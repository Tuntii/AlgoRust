use crate::indicators::{
    check_divergence, is_pivot_high, is_pivot_low, Adx, Atr, DivergenceType, Ema, Kama, Macd, Rsi,
    StochRsi,
};
use crate::order_block::OrderBlockTracker;
use crate::policy::BootstrapState;
use crate::types::{Candle, ContextId, MarketStructure, SignalType, TradeSignal, TrendState};
use chrono::Datelike;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::collections::VecDeque;

const SUPERKAMA_LENGTH: usize = 2584;
const SUPERKAMA_FAST_PERIOD: usize = 55;  // Pine: kama_fast = 55
const SUPERKAMA_SLOW_PERIOD: usize = 89;  // Pine: kama_slow = 89
const SUPERKAMA_ATR_PERIOD: usize = 33;
const MIN_CANDLE_BUFFER: usize = 30_000;
const SCALPER_REACTION_LOOKBACK: usize = 300;
const SCALPER_REACTION_K: usize = 4;
const SCALPER_REACTION_TOL_ATR_X: &str = "0.35";
const SCALPER_REACTION_MIN_TOUCHES: usize = 2;

#[derive(Debug, Clone)]
pub struct ScalperParams {
    pub ema_fast_period: usize,
    pub ema_slow_period: usize,
    pub pullback_atr_x: Decimal,
    pub rsi_long_min: u32,
    pub rsi_short_max: u32,
    pub swing_pivot_len: usize,
    pub ob_lookback: usize,
    pub ob_buffer_atr_x: Decimal,
    pub tp1_rr: Decimal,
    pub tp2_rr: Decimal,
}

impl ScalperParams {
    pub fn baseline() -> Self {
        Self {
            ema_fast_period: 9,
            ema_slow_period: 21,
            pullback_atr_x: Decimal::from_str_exact("0.35").unwrap(),
            rsi_long_min: 52,
            rsi_short_max: 48,
            swing_pivot_len: 3,
            ob_lookback: 20,
            ob_buffer_atr_x: Decimal::from_str_exact("0.15").unwrap(),
            tp1_rr: Decimal::from_str_exact("1.0").unwrap(),
            tp2_rr: Decimal::from_str_exact("1.8").unwrap(),
        }
    }

    pub fn hybrid() -> Self {
        Self {
            ema_fast_period: 12,
            ema_slow_period: 30,
            pullback_atr_x: Decimal::from_str_exact("0.40").unwrap(),
            rsi_long_min: 54,
            rsi_short_max: 46,
            swing_pivot_len: 3,
            ob_lookback: 20,
            ob_buffer_atr_x: Decimal::from_str_exact("0.15").unwrap(),
            tp1_rr: Decimal::from_str_exact("1.0").unwrap(),
            tp2_rr: Decimal::from_str_exact("1.8").unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TradeLevels {
    pub entry: Decimal,
    pub sl: Decimal,
    pub tp1: Decimal,
    pub tp2: Decimal,
}

pub struct SymbolContext {
    pub symbol: String,
    pub timeframe: String,
    pub scalper_params: ScalperParams,
    pub candles: VecDeque<Candle>,
    pub structure: MarketStructure,

    // Indicators
    pub ema_9: Ema,
    pub ema_21: Ema,
    pub ema_5: Ema,
    pub ema_8: Ema,
    pub ema_13: Ema,
    pub ema_50: Ema,
    pub ema_200: Ema,

    pub atr_14: Atr,
    pub rsi_14: Rsi, // RSI Indicator
    pub superkama_atr: Atr,

    // Pine strategy indicators (15m optimized)
    pub kama_10: Kama,
    pub macd_8_21_5: Macd,
    pub adx_10_10: Adx,
    pub atr_10: Atr,
    pub rsi_10: Rsi,
    pub stoch_rsi_10: StochRsi,

    pub kama_10_series: VecDeque<Decimal>,
    pub superkama_atr_series: VecDeque<Decimal>,
    pub rsi_10_series: VecDeque<Option<Decimal>>,
    pub stoch_k_series: VecDeque<Option<Decimal>>,

    pub pine_supertrend: Option<Decimal>,
    pub pine_trend: i32,
    pub pine_trend_changed_bullish: bool,
    pub pine_trend_changed_bearish: bool,
    pub pine_ema_cross_above: bool,
    pub pine_ema_cross_below: bool,
    pub pine_ema_above_kama: Option<bool>,
    pub pine_upper_band: Option<Decimal>,
    pub pine_lower_band: Option<Decimal>,
    pub pine_buy_signal: bool,
    pub pine_sell_signal: bool,
    pub pine_strong_buy: bool,
    pub pine_strong_sell: bool,
    pub pine_kama_long_filter: bool,
    pub pine_kama_short_filter: bool,
    pub pine_kama_quality_score: i32,
    pub pine_kama_slope_norm: Option<Decimal>,
    /// KAMA Efficiency Ratio — 0.0 = choppy, 1.0 = strong trend
    pub pine_kama_er: Decimal,

    pub pine_bullish_div: bool,
    pub pine_bearish_div: bool,
    pub pine_rsi_bullish_div: bool,
    pub pine_rsi_bearish_div: bool,
    pub pine_stoch_bullish_div: bool,
    pub pine_stoch_bearish_div: bool,

    // Session VWAP + attached Pine scalper state
    pub vwap_current: Option<Decimal>,
    vwap_session_pv: Decimal,
    vwap_session_volume: Decimal,
    vwap_session_day_key: Option<i32>,
    pub scalp_bull_trend: bool,
    pub scalp_bear_trend: bool,
    pub scalp_near_vwap: bool,
    pub scalp_mom_long: bool,
    pub scalp_mom_short: bool,
    pub scalp_long_ob_ok: bool,
    pub scalp_short_ob_ok: bool,
    pub scalp_long_signal: bool,
    pub scalp_short_signal: bool,
    pub scalp_bull_ob_top: Option<Decimal>,
    pub scalp_bull_ob_bottom: Option<Decimal>,
    pub scalp_bear_ob_top: Option<Decimal>,
    pub scalp_bear_ob_bottom: Option<Decimal>,

    pub atr_ratio_history: VecDeque<Decimal>, // Median ATR hesaplamak için tarihçe
    pub ema_50_slope_history: VecDeque<Decimal>, // EMA50 tarihçesi eğim hesabı için
    pub rsi_history: VecDeque<(usize, Decimal)>, // RSI history (index, value)

    // Events
    pub just_confirmed_pivot_high: bool,
    pub just_confirmed_pivot_low: bool,
    pub last_signal_candle: Option<usize>, // Son sinyal üretilen mum indeksi (cooldown için)
    /// Pending signal: mum kapanışında tespit edildi, bir sonraki mum açılışında execute edilecek
    pub pending_signal: Option<TradeSignal>,

    // BOS/Liquidity tracking
    pub just_broke_high: bool,                 // Bu mumda high kırıldı mı?
    pub just_broke_low: bool,                  // Bu mumda low kırıldı mı?
    pub pivot_high_history: VecDeque<Decimal>, // Son pivot high'lar (equal high tespiti için)
    pub pivot_low_history: VecDeque<Decimal>,  // Son pivot low'lar (equal low tespiti için)

    // Pivot History with Index for Divergence
    pub pivot_highs_with_idx: Vec<(usize, Decimal)>,
    pub pivot_lows_with_idx: Vec<(usize, Decimal)>,
    pub current_divergence: DivergenceType,

    // T0.2 — Bootstrap Integrity Gate
    pub bootstrap: BootstrapState,

    // Backtest tracking - Total candles ever processed (not just in buffer)
    pub total_candles_processed: usize,

    // MULTI-POSITION: Current context ID for signal generation
    pub current_context_id: Option<ContextId>,
    // Last BOS candle index (for context generation)
    pub last_bos_candle_idx: Option<usize>,
    // Last pivot candle indices
    pub last_pivot_high_idx: Option<usize>,
    pub last_pivot_low_idx: Option<usize>,

    // ORDER BLOCK: Smart Money TP/SL sistemi
    pub ob_tracker: OrderBlockTracker,
}

impl SymbolContext {
    pub fn new(symbol: String, timeframe: String) -> Self {
        Self::new_with_params(symbol, timeframe, ScalperParams::baseline())
    }

    pub fn new_with_params(symbol: String, timeframe: String, scalper_params: ScalperParams) -> Self {
        let ema_fast_period = scalper_params.ema_fast_period;
        let ema_slow_period = scalper_params.ema_slow_period;
        Self {
            symbol,
            timeframe: timeframe.clone(),
            scalper_params,
            candles: VecDeque::new(),
            structure: MarketStructure::default(),
            ema_50_slope_history: VecDeque::new(),
            atr_ratio_history: VecDeque::new(),
            pivot_high_history: VecDeque::new(),
            pivot_low_history: VecDeque::new(),
            pivot_highs_with_idx: Vec::new(),
            pivot_lows_with_idx: Vec::new(),
            rsi_history: VecDeque::new(),
            ema_9: Ema::new(ema_fast_period),
            ema_21: Ema::new(ema_slow_period),
            ema_5: Ema::new(5),
            ema_8: Ema::new(8),
            ema_13: Ema::new(13),
            ema_50: Ema::new(50),
            ema_200: Ema::new(200),
            atr_14: Atr::new(14),
            rsi_14: Rsi::new(14),
            kama_10: Kama::new(
                SUPERKAMA_LENGTH,
                SUPERKAMA_FAST_PERIOD,
                SUPERKAMA_SLOW_PERIOD,
            ),
            macd_8_21_5: Macd::new(8, 21, 5),
            adx_10_10: Adx::new(10, 10),
            atr_10: Atr::new(10),
            rsi_10: Rsi::new(10),
            stoch_rsi_10: StochRsi::new(10, 3, 3),
            superkama_atr: Atr::new(SUPERKAMA_ATR_PERIOD),
            kama_10_series: VecDeque::new(),
            superkama_atr_series: VecDeque::new(),
            rsi_10_series: VecDeque::new(),
            stoch_k_series: VecDeque::new(),
            pine_supertrend: None,
            pine_trend: 1,
            pine_trend_changed_bullish: false,
            pine_trend_changed_bearish: false,
            pine_ema_cross_above: false,
            pine_ema_cross_below: false,
            pine_ema_above_kama: None,
            pine_upper_band: None,
            pine_lower_band: None,
            pine_buy_signal: false,
            pine_sell_signal: false,
            pine_strong_buy: false,
            pine_strong_sell: false,
            pine_kama_long_filter: false,
            pine_kama_short_filter: false,
            pine_kama_quality_score: 0,
            pine_kama_slope_norm: None,
            pine_kama_er: Decimal::ZERO,
            pine_bullish_div: false,
            pine_bearish_div: false,
            pine_rsi_bullish_div: false,
            pine_rsi_bearish_div: false,
            pine_stoch_bullish_div: false,
            pine_stoch_bearish_div: false,
            vwap_current: None,
            vwap_session_pv: Decimal::ZERO,
            vwap_session_volume: Decimal::ZERO,
            vwap_session_day_key: None,
            scalp_bull_trend: false,
            scalp_bear_trend: false,
            scalp_near_vwap: false,
            scalp_mom_long: false,
            scalp_mom_short: false,
            scalp_long_ob_ok: false,
            scalp_short_ob_ok: false,
            scalp_long_signal: false,
            scalp_short_signal: false,
            scalp_bull_ob_top: None,
            scalp_bull_ob_bottom: None,
            scalp_bear_ob_top: None,
            scalp_bear_ob_bottom: None,
            just_confirmed_pivot_high: false,
            just_confirmed_pivot_low: false,
            current_divergence: DivergenceType::None,
            last_signal_candle: None,
            pending_signal: None,
            just_broke_high: false,
            just_broke_low: false,
            bootstrap: BootstrapState::with_timeframe(&timeframe), // TF-aware bootstrap
            total_candles_processed: 0,
            // Multi-position fields
            current_context_id: None,
            last_bos_candle_idx: None,
            last_pivot_high_idx: None,
            last_pivot_low_idx: None,

            // Order Block tracker
            ob_tracker: OrderBlockTracker::new(),
        }
    }

    /// Generate a context ID for the current signal opportunity
    /// This is used for multi-position uniqueness checking
    pub fn generate_context_id(&self) -> ContextId {
        let candle_idx = self.total_candles_processed;

        // Priority: BOS > Liquidity Sweep > Pivot
        if self.structure.bos_confirmed && self.just_broke_high {
            ContextId::from_bos(candle_idx, true)
        } else if self.structure.bos_confirmed && self.just_broke_low {
            ContextId::from_bos(candle_idx, false)
        } else if self.structure.has_equal_lows && self.just_confirmed_pivot_low {
            if let Some(pivot_val) = self.structure.last_pivot_low {
                ContextId::from_liquidity_sweep(candle_idx, pivot_val)
            } else {
                ContextId::from_pivot(candle_idx, false)
            }
        } else if self.structure.has_equal_highs && self.just_confirmed_pivot_high {
            if let Some(pivot_val) = self.structure.last_pivot_high {
                ContextId::from_liquidity_sweep(candle_idx, pivot_val)
            } else {
                ContextId::from_pivot(candle_idx, true)
            }
        } else if self.just_confirmed_pivot_low {
            ContextId::from_pivot(candle_idx, false)
        } else if self.just_confirmed_pivot_high {
            ContextId::from_pivot(candle_idx, true)
        } else {
            // Fallback: use candle index as identifier
            ContextId::new("signal", &candle_idx.to_string(), candle_idx)
        }
    }

    pub fn get_median_atr_ratio(&self) -> Decimal {
        if self.atr_ratio_history.is_empty() {
            return Decimal::ZERO;
        }
        let mut sorted: Vec<Decimal> = self.atr_ratio_history.iter().cloned().collect();
        // sort_by ile Decimal sıralama
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        sorted[mid]
    }

    // EMA50 Slope: 6-bar lookback for robust slope
    // (Current - EMA[t-6]) / EMA[t-6]
    pub fn get_ema50_slope(&self) -> Decimal {
        let len = self.ema_50_slope_history.len();
        if len < 7 {
            return Decimal::ZERO;
        }
        let current = self.ema_50_slope_history.back().unwrap();
        let old = self.ema_50_slope_history.get(len - 7).unwrap();

        if old.is_zero() {
            return Decimal::ZERO;
        }
        (*current - *old) / *old
    }

    pub fn add_candle(&mut self, candle: Candle) {
        // Increment total candles counter (for cooldown tracking)
        self.total_candles_processed += 1;

        // Reset events
        self.just_confirmed_pivot_high = false;
        self.just_confirmed_pivot_low = false;
        self.just_broke_high = false;
        self.just_broke_low = false;
        self.structure.last_bos_displacement = false;
        self.current_context_id = None; // Reset context for new candle

        self.pine_trend_changed_bullish = false;
        self.pine_trend_changed_bearish = false;
        self.pine_ema_cross_above = false;
        self.pine_ema_cross_below = false;
        self.pine_upper_band = None;
        self.pine_lower_band = None;
        self.pine_buy_signal = false;
        self.pine_sell_signal = false;
        self.pine_strong_buy = false;
        self.pine_strong_sell = false;
        self.pine_kama_long_filter = false;
        self.pine_kama_short_filter = false;
        self.pine_kama_quality_score = 0;
        self.pine_kama_slope_norm = None;
        self.pine_bullish_div = false;
        self.pine_bearish_div = false;
        self.pine_rsi_bullish_div = false;
        self.pine_rsi_bearish_div = false;
        self.pine_stoch_bullish_div = false;
        self.pine_stoch_bearish_div = false;
        self.scalp_bull_trend = false;
        self.scalp_bear_trend = false;
        self.scalp_near_vwap = false;
        self.scalp_mom_long = false;
        self.scalp_mom_short = false;
        self.scalp_long_ob_ok = false;
        self.scalp_short_ob_ok = false;
        self.scalp_long_signal = false;
        self.scalp_short_signal = false;

        // Update EMAs
        let close = candle.close;
        self.ema_9.update(close);
        self.ema_21.update(close);
        self.ema_5.update(close);
        self.ema_8.update(close);
        self.ema_13.update(close);
        let cur_ema50 = self.ema_50.update(close);
        self.ema_200.update(close);

        let session_day_key = candle.open_time.date_naive().num_days_from_ce();
        if self.vwap_session_day_key != Some(session_day_key) {
            self.vwap_session_day_key = Some(session_day_key);
            self.vwap_session_pv = Decimal::ZERO;
            self.vwap_session_volume = Decimal::ZERO;
        }
        let typical_price = (candle.high + candle.low + candle.close) / Decimal::from(3);
        self.vwap_session_pv += typical_price * candle.volume;
        self.vwap_session_volume += candle.volume;
        self.vwap_current = if self.vwap_session_volume.is_zero() {
            None
        } else {
            Some(self.vwap_session_pv / self.vwap_session_volume)
        };

        // Pine indicators
        let cur_kama = self.kama_10.update(close);
        self.kama_10_series.push_back(cur_kama);
        if self.kama_10_series.len() > 2000 {
            self.kama_10_series.pop_front();
        }
        let _ = self.macd_8_21_5.update(close);
        let _ = self.adx_10_10.update(candle.high, candle.low, candle.close);
        let _ = self.atr_10.update(candle.high, candle.low, candle.close);
        let rsi_10_val = self.rsi_10.update(candle.close);
        self.rsi_10_series.push_back(rsi_10_val);
        if self.rsi_10_series.len() > 2000 {
            self.rsi_10_series.pop_front();
        }
        if let Some(sk_atr) = self
            .superkama_atr
            .update(candle.high, candle.low, candle.close)
        {
            self.superkama_atr_series.push_back(sk_atr);
            if self.superkama_atr_series.len() > 2000 {
                self.superkama_atr_series.pop_front();
            }
        }

        let stoch_update = if let Some(rsi_val) = rsi_10_val {
            self.stoch_rsi_10.update(rsi_val)
        } else {
            None
        };
        let stoch_k_val = stoch_update.map(|(k, _)| k);
        self.stoch_k_series.push_back(stoch_k_val);
        if self.stoch_k_series.len() > 2000 {
            self.stoch_k_series.pop_front();
        }

        // Update ATR
        if let Some(atr_val) = self.atr_14.update(candle.high, candle.low, candle.close) {
            if !candle.close.is_zero() {
                let ratio = atr_val / candle.close;
                self.atr_ratio_history.push_back(ratio);
                if self.atr_ratio_history.len() > 200 {
                    self.atr_ratio_history.pop_front();
                }
            }
        }

        // Track EMA50 history - EMA update() returns Decimal directly
        self.ema_50_slope_history.push_back(cur_ema50);
        if self.ema_50_slope_history.len() > 20 {
            self.ema_50_slope_history.pop_front();
        }

        // Update RSI
        if let Some(rsi_val) = self.rsi_14.update(candle.close) {
            self.rsi_history
                .push_back((self.total_candles_processed, rsi_val));
            // Keep history manageable (e.g., last 300)
            if self.rsi_history.len() > 300 {
                self.rsi_history.pop_front();
            }
        }

        let prev_close = self.candles.back().map(|c| c.close);

        // Store Candle - HTF needs more history
        self.candles.push_back(candle);
        let max_candles =
            BootstrapState::min_candles_for_tf(&self.timeframe).max(MIN_CANDLE_BUFFER);
        if self.candles.len() > max_candles {
            self.candles.pop_front();
        }

        // BOS Detection: attached Pine uses real close crossover/crossunder
        let active_candle = self.candles.back().cloned().unwrap();
        let candle_range = active_candle.high - active_candle.low;
        let atr = self.atr_14.current_value.unwrap_or(Decimal::ONE);
        let displacement_threshold = atr * Decimal::from_f64(1.2).unwrap();

        let bos_up = if let (Some(prev_high), Some(prev_close)) = (self.structure.last_pivot_high, prev_close) {
            prev_close <= prev_high && active_candle.close > prev_high
        } else {
            false
        };
        let bos_down = if let (Some(prev_low), Some(prev_close)) = (self.structure.last_pivot_low, prev_close) {
            prev_close >= prev_low && active_candle.close < prev_low
        } else {
            false
        };

        self.just_broke_high = bos_up;
        self.just_broke_low = bos_down;
        self.structure.bos_confirmed = bos_up || bos_down;
        if bos_up || bos_down {
            self.structure.bos_candle_range = Some(candle_range);
            self.structure.last_bos_displacement = candle_range > displacement_threshold;
            self.last_bos_candle_idx = Some(self.total_candles_processed);
        }

        self.update_scalper_order_blocks(bos_up, bos_down);

        // ORDER BLOCK: mevcut tracker'ı da güncel tut (legacy TP/SL / analiz uyumu)
        self.ob_tracker.update(
            &active_candle,
            self.total_candles_processed,
            self.atr_14.current_value,
            bos_up,
            bos_down,
        );

        self.update_structure();

        // Attached Pine scalper state (EMA/VWAP/RSI/OB)
        self.update_pine_state(close, cur_kama);

        // Expose KAMA Efficiency Ratio for quality gate
        self.pine_kama_er = self.kama_10.er;

        // T0.2 — Update Bootstrap State (TF-aware)
        let pivot_count = self
            .pivot_high_history
            .len()
            .min(self.pivot_low_history.len());
        self.bootstrap.update_with_tf(
            &self.timeframe,
            self.candles.len(),
            self.ema_200.current_value.is_some(),
            pivot_count,
            self.atr_14.current_value.is_some(),
        );
    }

    fn update_structure(&mut self) {
        let pivot_len = self.scalper_params.swing_pivot_len;

        if self.candles.len() < (pivot_len * 2 + 1) {
            return;
        }

        let idx = self
            .candles
            .len()
            .saturating_sub(pivot_len + 1);
        // Index conversion to total_processed (approximate for history, precise for current candle)
        // Correct index relative to history start is tricky with popping.
        // Better to use total_candles_processed offset.
        // Pivot detected at Candle[idx]. Wait, `idx` is index in `self.candles`.
        // The real candle index is `total_candles_processed - (self.candles.len() - idx) + 1`?
        // Let's simplify:
        // `total_candles_processed` is the index of the JUST ADDED candle (last one).
        // `idx` is `len - 4`. So it is 3 candles ago properly.
        let pivot_real_idx = self.total_candles_processed - pivot_len;

        if idx < pivot_len {
            return;
        }

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        if is_pivot_high(&highs, idx, pivot_len) {
            let pivot_val = highs[idx];
            self.structure.last_pivot_high = Some(pivot_val);
            self.just_confirmed_pivot_high = true;
            self.last_pivot_high_idx = Some(pivot_real_idx); // Track pivot index

            // Pivot history'e ekle (equal high tespiti için)
            self.pivot_high_history.push_back(pivot_val);
            if self.pivot_high_history.len() > 5 {
                self.pivot_high_history.pop_front();
            }

            self.pivot_highs_with_idx.push((pivot_real_idx, pivot_val));
            if self.pivot_highs_with_idx.len() > 10 {
                self.pivot_highs_with_idx.remove(0);
            }

            // Equal High kontrolü: Son 5 pivot high içinde %0.15 toleransla eşit var mı?
            self.structure.has_equal_highs =
                self.check_equal_levels(&self.pivot_high_history.clone());

            // Check Bearish Divergence (HH price, LH RSI)
            let rsi_vec: Vec<_> = self.rsi_history.iter().cloned().collect();
            let divergence = check_divergence(
                &self.pivot_highs_with_idx,
                &self.pivot_lows_with_idx,
                &rsi_vec,
            );
            if divergence == DivergenceType::Bearish {
                self.current_divergence = DivergenceType::Bearish;
            }
        }

        if is_pivot_low(&lows, idx, pivot_len) {
            let pivot_val = lows[idx];
            self.structure.last_pivot_low = Some(pivot_val);
            self.just_confirmed_pivot_low = true;
            self.last_pivot_low_idx = Some(pivot_real_idx); // Track pivot index

            // Pivot history'e ekle
            self.pivot_low_history.push_back(pivot_val);
            if self.pivot_low_history.len() > 5 {
                self.pivot_low_history.pop_front();
            }

            self.pivot_lows_with_idx.push((pivot_real_idx, pivot_val));
            if self.pivot_lows_with_idx.len() > 10 {
                self.pivot_lows_with_idx.remove(0);
            }

            // Equal Low kontrolü
            self.structure.has_equal_lows =
                self.check_equal_levels(&self.pivot_low_history.clone());

            // Check Bullish Divergence (LL price, HL RSI)
            let rsi_vec: Vec<_> = self.rsi_history.iter().cloned().collect();
            let divergence = check_divergence(
                &self.pivot_highs_with_idx,
                &self.pivot_lows_with_idx,
                &rsi_vec,
            );
            if divergence == DivergenceType::Bullish {
                self.current_divergence = DivergenceType::Bullish;
            }
        }

        // Update Trend
        if let (Some(fast), Some(slow)) = (self.ema_9.current_value, self.ema_21.current_value) {
            if fast > slow {
                self.structure.trend = TrendState::Bullish;
            } else if fast < slow {
                self.structure.trend = TrendState::Bearish;
            } else {
                self.structure.trend = TrendState::Neutral;
            }
        }
    }

    fn update_pine_state(&mut self, close: Decimal, kama: Decimal) {
        let atr = match self.atr_14.current_value {
            Some(val) => val,
            None => return,
        };
        let vwap = match self.vwap_current {
            Some(val) => val,
            None => return,
        };
        let ema_fast = match self.ema_9.current_value {
            Some(val) => val,
            None => return,
        };
        let ema_slow = match self.ema_21.current_value {
            Some(val) => val,
            None => return,
        };
        let rsi = match self.rsi_14.current_value {
            Some(val) => val,
            None => return,
        };
        let last_open = self.candles.back().map(|c| c.open).unwrap_or(close);

        let pullback_atr = self.scalper_params.pullback_atr_x;
        let bull_trend = ema_fast > ema_slow;
        let bear_trend = ema_fast < ema_slow;
        let near_vwap = (close - vwap).abs() <= atr * pullback_atr;
        let mom_long = rsi >= Decimal::from(self.scalper_params.rsi_long_min);
        let mom_short = rsi <= Decimal::from(self.scalper_params.rsi_short_max);

        let in_bull_ob = matches!((self.scalp_bull_ob_bottom, self.scalp_bull_ob_top),
            (Some(bottom), Some(top)) if close >= bottom && close <= top);
        let above_bull_ob = matches!(self.scalp_bull_ob_top, Some(top) if close > top);
        let in_bear_ob = matches!((self.scalp_bear_ob_bottom, self.scalp_bear_ob_top),
            (Some(bottom), Some(top)) if close >= bottom && close <= top);
        let below_bear_ob = matches!(self.scalp_bear_ob_bottom, Some(bottom) if close < bottom);

        let long_ob_ok = in_bull_ob || above_bull_ob;
        let short_ob_ok = in_bear_ob || below_bear_ob;
        let long_signal = bull_trend && near_vwap && close > last_open && mom_long && long_ob_ok;
        let short_signal = bear_trend && near_vwap && close < last_open && mom_short && short_ob_ok;

        let prev_trend = self.pine_trend;
        self.pine_trend = if bull_trend {
            1
        } else if bear_trend {
            -1
        } else {
            0
        };
        self.pine_trend_changed_bullish = self.pine_trend == 1 && prev_trend != 1;
        self.pine_trend_changed_bearish = self.pine_trend == -1 && prev_trend != -1;

        self.scalp_bull_trend = bull_trend;
        self.scalp_bear_trend = bear_trend;
        self.scalp_near_vwap = near_vwap;
        self.scalp_mom_long = mom_long;
        self.scalp_mom_short = mom_short;
        self.scalp_long_ob_ok = long_ob_ok;
        self.scalp_short_ob_ok = short_ob_ok;
        self.scalp_long_signal = long_signal;
        self.scalp_short_signal = short_signal;

        self.pine_supertrend = Some(vwap);
        self.pine_upper_band = self.scalp_bear_ob_top;
        self.pine_lower_band = self.scalp_bull_ob_bottom;
        self.pine_buy_signal = long_signal;
        self.pine_sell_signal = short_signal;
        self.pine_strong_buy = false;
        self.pine_strong_sell = false;

        self.pine_ema_cross_above = self.pine_trend_changed_bullish;
        self.pine_ema_cross_below = self.pine_trend_changed_bearish;
        self.pine_ema_above_kama = Some(bull_trend);
        self.pine_kama_long_filter = bull_trend;
        self.pine_kama_short_filter = bear_trend;
        self.pine_kama_slope_norm = if atr.is_zero() {
            Some(Decimal::ZERO)
        } else {
            Some((ema_fast - ema_slow) / atr)
        };
        self.pine_kama_quality_score = if long_signal {
            1
        } else if short_signal {
            -1
        } else {
            0
        };

        // This indicator does not include divergence logic.
        self.pine_bullish_div = false;
        self.pine_bearish_div = false;
        self.pine_rsi_bullish_div = false;
        self.pine_rsi_bearish_div = false;
        self.pine_stoch_bullish_div = false;
        self.pine_stoch_bearish_div = false;

        let _ = kama;
    }

    /// Equal high/low tespiti: Son pivot'lar arasında %0.15 toleransla eşit seviye var mı?
    fn check_equal_levels(&self, pivots: &VecDeque<Decimal>) -> bool {
        if pivots.len() < 2 {
            return false;
        }

        let tolerance = Decimal::from_f64(0.0015).unwrap(); // %0.15

        for i in 0..pivots.len() {
            for j in (i + 1)..pivots.len() {
                let p1 = pivots[i];
                let p2 = pivots[j];
                if p1.is_zero() {
                    continue;
                }

                let diff_pct = ((p1 - p2) / p1).abs();
                if diff_pct < tolerance {
                    return true; // Equal level bulundu = Liquidity pool
                }
            }
        }
        false
    }

    fn update_scalper_order_blocks(&mut self, bos_up: bool, bos_down: bool) {
        let ob_lookback = self.scalper_params.ob_lookback;

        if bos_up {
            if let Some(found) = self
                .candles
                .iter()
                .rev()
                .skip(1)
                .take(ob_lookback)
                .find(|c| c.close < c.open)
            {
                self.scalp_bull_ob_top = Some(found.open);
                self.scalp_bull_ob_bottom = Some(found.low);
            }
        }

        if bos_down {
            if let Some(found) = self
                .candles
                .iter()
                .rev()
                .skip(1)
                .take(ob_lookback)
                .find(|c| c.close > c.open)
            {
                self.scalp_bear_ob_top = Some(found.high);
                self.scalp_bear_ob_bottom = Some(found.open);
            }
        }
    }

    fn reaction_avg(&self, use_highs: bool, need_above_entry: bool, entry: Decimal) -> Option<Decimal> {
        let atr = self.atr_14.current_value?;
        let tolerance = atr * Decimal::from_str_exact(SCALPER_REACTION_TOL_ATR_X).unwrap();
        let min_idx = self
            .total_candles_processed
            .saturating_sub(SCALPER_REACTION_LOOKBACK);
        let points = if use_highs {
            &self.pivot_highs_with_idx
        } else {
            &self.pivot_lows_with_idx
        };

        let mut cluster_levels: Vec<Decimal> = Vec::new();
        let mut cluster_counts: Vec<usize> = Vec::new();

        for (idx, price) in points.iter().copied() {
            if idx < min_idx {
                continue;
            }

            let mut clustered = false;
            for cluster_idx in 0..cluster_levels.len() {
                let level = cluster_levels[cluster_idx];
                if (price - level).abs() <= tolerance {
                    let count = cluster_counts[cluster_idx];
                    let new_level = (level * Decimal::from(count as u32) + price)
                        / Decimal::from((count + 1) as u32);
                    cluster_levels[cluster_idx] = new_level;
                    cluster_counts[cluster_idx] = count + 1;
                    clustered = true;
                    break;
                }
            }

            if !clustered {
                cluster_levels.push(price);
                cluster_counts.push(1);
            }
        }

        let mut candidates: Vec<(Decimal, usize)> = cluster_levels
            .into_iter()
            .zip(cluster_counts.into_iter())
            .filter(|(level, count)| {
                let side_ok = if need_above_entry {
                    *level > entry
                } else {
                    *level < entry
                };
                side_ok && *count >= SCALPER_REACTION_MIN_TOUCHES
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by(|(level_a, count_a), (level_b, count_b)| {
            count_b.cmp(count_a).then_with(|| {
                let dist_a = (*level_a - entry).abs();
                let dist_b = (*level_b - entry).abs();
                dist_a.cmp(&dist_b)
            })
        });

        let selected = candidates.into_iter().take(SCALPER_REACTION_K).collect::<Vec<_>>();
        if selected.is_empty() {
            return None;
        }

        let sum: Decimal = selected.iter().map(|(level, _)| *level).sum();
        Some(sum / Decimal::from(selected.len() as u32))
    }

    fn safe_risk(&self, entry: Decimal, sl: Option<Decimal>, is_long: bool) -> Decimal {
        let atr = self.atr_14.current_value.unwrap_or(Decimal::ONE);
        let fallback_step = (atr / Decimal::from(2)).max(Decimal::from_str_exact("0.00000001").unwrap());
        let mut out = sl.unwrap_or(if is_long { entry - atr } else { entry + atr });

        if is_long && out >= entry {
            out = entry - fallback_step;
        }
        if !is_long && out <= entry {
            out = entry + fallback_step;
        }
        out
    }

    fn safe_tp(&self, entry: Decimal, tp: Option<Decimal>, is_long: bool, fallback_r: Decimal, rr_mult: Decimal) -> Decimal {
        let mut out = tp.unwrap_or(if is_long {
            entry + fallback_r * rr_mult
        } else {
            entry - fallback_r * rr_mult
        });

        if is_long && out <= entry {
            out = entry + fallback_r * rr_mult;
        }
        if !is_long && out >= entry {
            out = entry - fallback_r * rr_mult;
        }
        out
    }

    pub fn calculate_trade_levels(&self, direction: &SignalType, entry: Decimal) -> TradeLevels {
        let atr = self.atr_14.current_value.unwrap_or(Decimal::ONE);
        let buffer = atr * self.scalper_params.ob_buffer_atr_x;
        let tp1_rr = self.scalper_params.tp1_rr;
        let tp2_rr = self.scalper_params.tp2_rr;
        let min_r = Decimal::from_str_exact("0.00000001").unwrap();

        match direction {
            SignalType::LONG => {
                let sl_react = self.reaction_avg(false, false, entry);
                let tp_react = self.reaction_avg(true, true, entry);
                let structural_sl = self
                    .scalp_bull_ob_bottom
                    .map(|bottom| bottom - buffer)
                    .or_else(|| self.structure.last_pivot_low)
                    .or(Some(entry - atr));
                let sl = self.safe_risk(entry, sl_react.map(|v| v - buffer).or(structural_sl), true);

                let risk = (entry - sl).max(min_r);
                let swing_tp1 = self.structure.last_pivot_high.unwrap_or(entry + risk * tp1_rr);
                let tp1 = self.safe_tp(entry, tp_react.or(Some(swing_tp1)), true, risk, tp1_rr);
                let tp2 = self.safe_tp(entry, Some(entry + risk * tp2_rr), true, risk, tp2_rr);

                TradeLevels { entry, sl, tp1, tp2 }
            }
            SignalType::SHORT => {
                let sl_react = self.reaction_avg(true, true, entry);
                let tp_react = self.reaction_avg(false, false, entry);
                let structural_sl = self
                    .scalp_bear_ob_top
                    .map(|top| top + buffer)
                    .or_else(|| self.structure.last_pivot_high)
                    .or(Some(entry + atr));
                let sl = self.safe_risk(entry, sl_react.map(|v| v + buffer).or(structural_sl), false);

                let risk = (sl - entry).max(min_r);
                let swing_tp1 = self.structure.last_pivot_low.unwrap_or(entry - risk * tp1_rr);
                let tp1 = self.safe_tp(entry, tp_react.or(Some(swing_tp1)), false, risk, tp1_rr);
                let tp2 = self.safe_tp(entry, Some(entry - risk * tp2_rr), false, risk, tp2_rr);

                TradeLevels { entry, sl, tp1, tp2 }
            }
        }
    }

    pub fn indicator_reversal_for(&self, direction: &SignalType) -> bool {
        match direction {
            SignalType::LONG => self.scalp_short_signal,
            SignalType::SHORT => self.scalp_long_signal,
        }
    }
}
