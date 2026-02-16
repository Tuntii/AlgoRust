use crate::state::SymbolContext;
use anyhow::{anyhow, Result};
use chrono::Timelike;
use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;
use rust_decimal::prelude::ToPrimitive;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::path::Path;
use std::sync::{Arc, Mutex};

const FEATURE_COUNT: usize = 15;

#[derive(Debug, Deserialize)]
struct MetaFile {
    lookback: usize,
    threshold: f32,
    mean: JsonValue,
    std: JsonValue,
}

#[derive(Clone)]
pub struct LstmFilter {
    session: Arc<Mutex<Session>>,
    lookback: usize,
    threshold: f32,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl LstmFilter {
    pub fn load(model_path: &str, meta_path: &str, ort_path: Option<&str>) -> Result<Self> {
        let meta_text = std::fs::read_to_string(meta_path)?;
        let meta: MetaFile = serde_json::from_str(&meta_text)?;
        let mean = flatten_meta(&meta.mean)?;
        let std = flatten_meta(&meta.std)?;

        if mean.len() != FEATURE_COUNT || std.len() != FEATURE_COUNT {
            return Err(anyhow!(
                "Meta mean/std size mismatch: {} / {}",
                mean.len(),
                std.len()
            ));
        }

        if let Some(path) = ort_path {
            if !path.trim().is_empty() {
                let _ = ort::init_from(Path::new(path))?.commit();
            } else {
                let _ = ort::init().commit();
            }
        } else {
            let _ = ort::init().commit();
        }
        let session = Session::builder()?.commit_from_file(Path::new(model_path))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            lookback: meta.lookback,
            threshold: meta.threshold,
            mean,
            std,
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn score(&self, ctx: &SymbolContext) -> Result<Option<f32>> {
        let features = build_feature_window(ctx, self.lookback, &self.mean, &self.std)?;
        let Some(features) = features else {
            return Ok(None);
        };

        let input = Array3::from_shape_vec((1, self.lookback, FEATURE_COUNT), features)?;
        let shape = input.raw_dim();
        let data = input.into_raw_vec();
        let dims = vec![shape[0], shape[1], shape[2]];
        let input_tensor = Tensor::from_array((dims, data))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;
        let output = outputs["output"].try_extract_array::<f32>()?;
        let score = output.iter().next().copied().unwrap_or(0.0);

        Ok(Some(score))
    }

}

fn flatten_meta(value: &JsonValue) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    flatten_value(value, &mut out)?;
    Ok(out)
}

fn flatten_value(value: &JsonValue, out: &mut Vec<f32>) -> Result<()> {
    match value {
        JsonValue::Array(items) => {
            for item in items {
                flatten_value(item, out)?;
            }
            Ok(())
        }
        JsonValue::Number(num) => {
            let v: f64 = num
                .as_f64()
                .ok_or_else(|| anyhow!("Invalid numeric value in meta"))?;
            out.push(v as f32);
            Ok(())
        }
        _ => Err(anyhow!("Invalid meta format")),
    }
}

fn build_feature_window(
    ctx: &SymbolContext,
    lookback: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<Option<Vec<f32>>> {
    if ctx.candles.len() < lookback + 1 {
        return Ok(None);
    }

    let candles: Vec<_> = ctx.candles.iter().collect();
    let n = candles.len();
    let start = n - lookback;

    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);

    for c in &candles {
        closes.push(c.close.to_f64().unwrap_or(0.0));
        highs.push(c.high.to_f64().unwrap_or(0.0));
        lows.push(c.low.to_f64().unwrap_or(0.0));
        volumes.push(c.volume.to_f64().unwrap_or(0.0));
        times.push(c.open_time);
    }

    let log_return = log_returns(&closes);
    let rolling_vol_20 = rolling_std(&log_return, 20);
    let atr_14 = atr_series(&highs, &lows, &closes, 14);
    let return_over_atr = return_over_atr(&log_return, &atr_14, &closes);

    let kama = kama_series(&closes, 10, 2, 20);
    let kama_slope = diff_over_close(&kama, &closes);
    let close_kama_over_atr = close_minus_kama_over_atr(&closes, &kama, &atr_14);

    let volume_zscore = rolling_zscore(&volumes, 20);
    let vwap_distance = daily_vwap_distance(&closes, &volumes, &times);
    let (time_sin, time_cos) = time_sin_cos(&times);

    // New V2 features
    let rsi_14 = rsi_series(&closes, 14);
    let macd_hist_norm = macd_hist_normalized(&closes, 12, 26, 9);
    let adx_norm = adx_series(&highs, &lows, &closes, 14);
    let bb_position = bollinger_position(&closes, 20, 2.0);

    let mut out = Vec::with_capacity(lookback * FEATURE_COUNT);
    for i in start..n {
        let close = closes[i];
        let close_safe = if close == 0.0 { 1.0 } else { close };

        let raw = [
            log_return[i],
            rolling_vol_20[i],
            atr_14[i],
            return_over_atr[i],
            if close_safe == 0.0 {
                0.0
            } else {
                kama[i] / close_safe
            },
            kama_slope[i],
            close_kama_over_atr[i],
            volume_zscore[i],
            vwap_distance[i],
            time_sin[i],
            time_cos[i],
            // V2 features
            rsi_14[i],
            macd_hist_norm[i],
            adx_norm[i],
            bb_position[i],
        ];

        for (j, v) in raw.iter().enumerate() {
            let denom = if std[j] == 0.0 { 1.0 } else { std[j] };
            let standardized = (*v as f32 - mean[j]) / denom;
            out.push(standardized);
        }
    }

    Ok(Some(out))
}

fn log_returns(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if i == 0 || values[i - 1] == 0.0 || values[i] == 0.0 {
            out.push(0.0);
        } else {
            out.push((values[i] / values[i - 1]).ln());
        }
    }
    out
}

fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let slice = &values[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        out.push(var.sqrt());
    }
    out
}

fn atr_series(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0; n];
    for i in 0..n {
        if i == 0 {
            tr[i] = highs[i] - lows[i];
        } else {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
    }

    let mut atr = vec![0.0; n];
    if n > 0 {
        for i in 0..n {
            let start = if i + 1 >= period { i + 1 - period } else { 0 };
            let sum: f64 = tr[start..=i].iter().sum();
            atr[i] = sum / (i + 1 - start) as f64;
        }
    }

    atr
}

fn return_over_atr(log_ret: &[f64], atr: &[f64], closes: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(log_ret.len());
    for i in 0..log_ret.len() {
        let close = closes[i];
        let atr_pct = if close == 0.0 { 0.0 } else { atr[i] / close };
        if atr_pct == 0.0 {
            out.push(0.0);
        } else {
            out.push(log_ret[i] / atr_pct);
        }
    }
    out
}

fn kama_series(closes: &[f64], length: usize, fast: usize, slow: usize) -> Vec<f64> {
    let n = closes.len();
    if n == 0 {
        return Vec::new();
    }

    let fast_sc = 2.0 / (fast as f64 + 1.0);
    let slow_sc = 2.0 / (slow as f64 + 1.0);
    let mut out = vec![closes[0]; n];
    let mut abs_changes = Vec::with_capacity(n);

    for i in 1..n {
        abs_changes.push((closes[i] - closes[i - 1]).abs());
        let start = if i >= length { i - length } else { 0 };
        let sum_abs: f64 = abs_changes[start..i].iter().sum();
        let change = (closes[i] - closes[start]).abs();
        let er = if sum_abs == 0.0 {
            0.0
        } else {
            change / sum_abs
        };
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
        out[i] = out[i - 1] + sc * (closes[i] - out[i - 1]);
    }

    out
}

fn diff_over_close(values: &[f64], closes: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if i == 0 || closes[i] == 0.0 {
            out.push(0.0);
        } else {
            out.push((values[i] - values[i - 1]) / closes[i]);
        }
    }
    out
}

fn close_minus_kama_over_atr(closes: &[f64], kama: &[f64], atr: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(closes.len());
    for i in 0..closes.len() {
        if atr[i] == 0.0 {
            out.push(0.0);
        } else {
            out.push((closes[i] - kama[i]) / atr[i]);
        }
    }
    out
}

fn rolling_zscore(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let slice = &values[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        let std = var.sqrt();
        if std == 0.0 {
            out.push(0.0);
        } else {
            out.push((values[i] - mean) / std);
        }
    }
    out
}

fn daily_vwap_distance(
    closes: &[f64],
    volumes: &[f64],
    times: &[chrono::DateTime<chrono::Utc>],
) -> Vec<f64> {
    let mut out = Vec::with_capacity(closes.len());
    let mut cum_pv = 0.0;
    let mut cum_vol = 0.0;
    let mut current_day = None;

    for i in 0..closes.len() {
        let day = times[i].date_naive();
        if current_day.map(|d| d != day).unwrap_or(true) {
            current_day = Some(day);
            cum_pv = 0.0;
            cum_vol = 0.0;
        }
        cum_pv += closes[i] * volumes[i];
        cum_vol += volumes[i];
        let vwap = if cum_vol == 0.0 {
            0.0
        } else {
            cum_pv / cum_vol
        };
        if vwap == 0.0 {
            out.push(0.0);
        } else {
            out.push((closes[i] - vwap) / vwap);
        }
    }

    out
}

fn time_sin_cos(times: &[chrono::DateTime<chrono::Utc>]) -> (Vec<f64>, Vec<f64>) {
    let mut sin_out = Vec::with_capacity(times.len());
    let mut cos_out = Vec::with_capacity(times.len());
    for t in times {
        let seconds = (t.hour() * 3600 + t.minute() * 60 + t.second()) as f64;
        let angle = 2.0 * std::f64::consts::PI * seconds / 86400.0;
        sin_out.push(angle.sin());
        cos_out.push(angle.cos());
    }
    (sin_out, cos_out)
}

// ===== V2 Feature Functions =====

/// Wilder's RSI, normalized to 0-1.
fn rsi_series(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut out = vec![0.5; n]; // default to neutral
    if n < 2 {
        return out;
    }
    let alpha = 1.0 / period as f64;
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..n {
        let change = closes[i] - closes[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        // EWM with alpha=1/period, adjust=False
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;

        let rsi = if avg_loss == 0.0 {
            1.0
        } else {
            let rs = avg_gain / avg_loss;
            1.0 - (1.0 / (1.0 + rs))
        };
        out[i] = rsi; // already 0-1
    }
    out
}

/// MACD histogram normalized by close price.
fn macd_hist_normalized(closes: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let n = closes.len();
    if n == 0 {
        return Vec::new();
    }
    let fast_alpha = 2.0 / (fast as f64 + 1.0);
    let slow_alpha = 2.0 / (slow as f64 + 1.0);
    let signal_alpha = 2.0 / (signal as f64 + 1.0);

    let mut ema_fast = closes[0];
    let mut ema_slow = closes[0];
    let mut macd_line = 0.0;
    let mut signal_line = 0.0;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        ema_fast = fast_alpha * closes[i] + (1.0 - fast_alpha) * ema_fast;
        ema_slow = slow_alpha * closes[i] + (1.0 - slow_alpha) * ema_slow;
        macd_line = ema_fast - ema_slow;
        signal_line = signal_alpha * macd_line + (1.0 - signal_alpha) * signal_line;
        let hist = macd_line - signal_line;
        let close = closes[i];
        if close == 0.0 {
            out.push(0.0);
        } else {
            out.push(hist / close);
        }
    }
    out
}

/// ADX (Average Directional Index), normalized to 0-1.
fn adx_series(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut out = vec![0.0; n];
    if n < 2 {
        return out;
    }
    let alpha = 1.0 / period as f64;
    let mut atr_sm = highs[0] - lows[0];
    let mut plus_dm_sm = 0.0;
    let mut minus_dm_sm = 0.0;
    let mut adx_sm = 0.0;

    for i in 1..n {
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];

        let plus_dm = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let minus_dm = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        let tr = {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            hl.max(hc).max(lc)
        };

        // Wilder's smoothing (EWM with alpha=1/period)
        atr_sm = alpha * tr + (1.0 - alpha) * atr_sm;
        plus_dm_sm = alpha * plus_dm + (1.0 - alpha) * plus_dm_sm;
        minus_dm_sm = alpha * minus_dm + (1.0 - alpha) * minus_dm_sm;

        if atr_sm == 0.0 {
            out[i] = 0.0;
            continue;
        }

        let pdi = 100.0 * plus_dm_sm / atr_sm;
        let mdi = 100.0 * minus_dm_sm / atr_sm;
        let denom = pdi + mdi;
        let dx = if denom == 0.0 {
            0.0
        } else {
            100.0 * (pdi - mdi).abs() / denom
        };

        adx_sm = alpha * dx + (1.0 - alpha) * adx_sm;
        out[i] = (adx_sm / 100.0).clamp(0.0, 1.0);
    }
    out
}

/// Bollinger Band %B: position within bands.
fn bollinger_position(closes: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let n = closes.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let slice = &closes[start..=i];
        let len = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / len;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / len;
        let std = var.sqrt();
        let upper = mean + num_std * std;
        let lower = mean - num_std * std;
        let band_w = upper - lower;
        if band_w == 0.0 {
            out.push(0.5);
        } else {
            let pct = (closes[i] - lower) / band_w;
            out.push(pct.clamp(-1.0, 2.0));
        }
    }
    out
}
