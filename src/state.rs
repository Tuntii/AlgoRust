use crate::indicators::{
    check_divergence, is_pivot_high, is_pivot_low, Adx, Atr, DivergenceType, Ema, Kama, Macd, Rsi,
    StochRsi,
};
use crate::order_block::OrderBlockTracker;
use crate::policy::BootstrapState;
use crate::types::{Candle, ContextId, MarketStructure, TradeSignal, TrendState};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::collections::VecDeque;

const SUPERKAMA_LENGTH: usize = 2584;
const SUPERKAMA_FAST_PERIOD: usize = 55;  // Pine: kama_fast = 55
const SUPERKAMA_SLOW_PERIOD: usize = 89;  // Pine: kama_slow = 89
const SUPERKAMA_ATR_PERIOD: usize = 33;
const MIN_CANDLE_BUFFER: usize = 30_000;

pub struct SymbolContext {
    pub symbol: String,
    pub timeframe: String,
    pub candles: VecDeque<Candle>,
    pub structure: MarketStructure,

    // Indicators
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
        Self {
            symbol,
            timeframe: timeframe.clone(),
            candles: VecDeque::new(),
            structure: MarketStructure::default(),
            ema_50_slope_history: VecDeque::new(),
            atr_ratio_history: VecDeque::new(),
            pivot_high_history: VecDeque::new(),
            pivot_low_history: VecDeque::new(),
            pivot_highs_with_idx: Vec::new(),
            pivot_lows_with_idx: Vec::new(),
            rsi_history: VecDeque::new(),
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

        // Update EMAs
        let close = candle.close;
        self.ema_5.update(close);
        self.ema_8.update(close);
        self.ema_13.update(close);
        let cur_ema50 = self.ema_50.update(close);
        self.ema_200.update(close);

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

        // BOS Detection: Check if current candle broke previous swing
        let candle_range = candle.high - candle.low;
        let atr = self.atr_14.current_value.unwrap_or(Decimal::ONE);
        let displacement_threshold = atr * Decimal::from_f64(1.2).unwrap(); // BOS candle > 1.2*ATR = displacement

        if let Some(prev_high) = self.structure.last_pivot_high {
            if candle.close > prev_high {
                self.just_broke_high = true;
                self.structure.bos_confirmed = true;
                self.structure.bos_candle_range = Some(candle_range);
                self.structure.last_bos_displacement = candle_range > displacement_threshold;
                self.last_bos_candle_idx = Some(self.total_candles_processed); // Track BOS index
            }
        }

        if let Some(prev_low) = self.structure.last_pivot_low {
            if candle.close < prev_low {
                self.just_broke_low = true;
                self.structure.bos_confirmed = true;
                self.structure.bos_candle_range = Some(candle_range);
                self.structure.last_bos_displacement = candle_range > displacement_threshold;
                self.last_bos_candle_idx = Some(self.total_candles_processed); // Track BOS index
            }
        }

        // ORDER BLOCK: BOS tespitinden sonra OB tracker'ı güncelle
        self.ob_tracker.update(
            &candle,
            self.total_candles_processed,
            self.atr_14.current_value,
            self.just_broke_high, // Yukarı BOS
            self.just_broke_low,  // Aşağı BOS
        );

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

        // Store Candle - HTF needs more history
        self.candles.push_back(candle);
        let max_candles =
            BootstrapState::min_candles_for_tf(&self.timeframe).max(MIN_CANDLE_BUFFER);
        if self.candles.len() > max_candles {
            self.candles.pop_front();
        }

        self.update_structure();

        // SuperKAMA state (trend + crossover + band signals)
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
        if self.candles.len() < 7 {
            return;
        }

        // Check Pivot at index len - 4
        let idx = self.candles.len().saturating_sub(4);
        // Index conversion to total_processed (approximate for history, precise for current candle)
        // Correct index relative to history start is tricky with popping.
        // Better to use total_candles_processed offset.
        // Pivot detected at Candle[idx]. Wait, `idx` is index in `self.candles`.
        // The real candle index is `total_candles_processed - (self.candles.len() - idx) + 1`?
        // Let's simplify:
        // `total_candles_processed` is the index of the JUST ADDED candle (last one).
        // `idx` is `len - 4`. So it is 3 candles ago properly.
        let pivot_real_idx = self.total_candles_processed - 3;

        if idx < 3 {
            return;
        }

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        if is_pivot_high(&highs, idx) {
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

        if is_pivot_low(&lows, idx) {
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
        let e5 = self.ema_5.current_value;
        let e8 = self.ema_8.current_value;
        let e13 = self.ema_13.current_value;
        let e50 = self.ema_50.current_value;
        let e200 = self.ema_200.current_value;

        let last_close = self.candles.back().map(|c| c.close).unwrap_or_default();

        if let (Some(v5), Some(v8), Some(v13), Some(v50), Some(v200)) = (e5, e8, e13, e50, e200) {
            if v5 > v8 && v8 > v13 && last_close > v50 && v50 > v200 {
                self.structure.trend = TrendState::Bullish;
            } else if v5 < v8 && v8 < v13 && last_close < v50 && v50 < v200 {
                self.structure.trend = TrendState::Bearish;
            } else {
                self.structure.trend = TrendState::Neutral;
            }
        }
    }

    fn update_pine_state(&mut self, close: Decimal, kama: Decimal) {
        // indicator.pine parity:
        // upper/lower bands = KAMA +/- ATR(33) * 1.0
        let atr = match self.superkama_atr.current_value {
            Some(val) => val,
            None => return,
        };
        // Pine: atr_multiplier = 2.5  (synced with indicator.pine)
        let atr_multiplier = Decimal::new(25, 1); // 2.5 = 25 × 10^-1
        let upper_band = kama + atr * atr_multiplier;
        let lower_band = kama - atr * atr_multiplier;

        self.pine_upper_band = Some(upper_band);
        self.pine_lower_band = Some(lower_band);
        // Keep legacy field populated for compatibility with existing report/exit plumbing.
        self.pine_supertrend = Some(kama);

        let prev_close = if self.candles.len() >= 2 {
            self.candles.get(self.candles.len() - 2).map(|c| c.close)
        } else {
            None
        };
        let prev_kama = if self.kama_10_series.len() >= 2 {
            self.kama_10_series
                .get(self.kama_10_series.len() - 2)
                .copied()
        } else {
            None
        };
        let prev_atr = if self.superkama_atr_series.len() >= 2 {
            self.superkama_atr_series
                .get(self.superkama_atr_series.len() - 2)
                .copied()
        } else {
            None
        };

        let kama_rising = prev_kama.as_ref().map(|prev| {
            kama > *prev
        }).unwrap_or(false);
        let kama_falling = prev_kama.as_ref().map(|prev| {
            kama < *prev
        }).unwrap_or(false);
        let price_above_kama = close > kama;
        let price_below_kama = close < kama;

        let bullish_trend = kama_rising && price_above_kama;
        let bearish_trend = kama_falling && price_below_kama;

        let prev_trend = self.pine_trend;
        self.pine_trend = if bullish_trend {
            1
        } else if bearish_trend {
            -1
        } else {
            0
        };
        self.pine_trend_changed_bullish = self.pine_trend == 1 && prev_trend != 1;
        self.pine_trend_changed_bearish = self.pine_trend == -1 && prev_trend != -1;

        let buy_signal =
            if let (Some(prev_c), Some(prev_k)) = (prev_close.as_ref(), prev_kama.as_ref()) {
                close > kama && *prev_c <= *prev_k
            } else {
                false
            };
        let sell_signal =
            if let (Some(prev_c), Some(prev_k)) = (prev_close.as_ref(), prev_kama.as_ref()) {
                close < kama && *prev_c >= *prev_k
            } else {
                false
            };

        let (strong_buy, strong_sell) = if let (Some(prev_c), Some(prev_k), Some(prev_a)) =
            (prev_close.as_ref(), prev_kama.as_ref(), prev_atr.as_ref())
        {
            let prev_upper = *prev_k + *prev_a * atr_multiplier;
            let prev_lower = *prev_k - *prev_a * atr_multiplier;
            (
                close > lower_band && *prev_c <= prev_lower,
                close < upper_band && *prev_c >= prev_upper,
            )
        } else {
            (false, false)
        };

        self.pine_buy_signal = buy_signal;
        self.pine_sell_signal = sell_signal;
        self.pine_strong_buy = strong_buy;
        self.pine_strong_sell = strong_sell;

        // Legacy aliases consumed by existing engine wiring.
        self.pine_ema_cross_above = buy_signal;
        self.pine_ema_cross_below = sell_signal;
        self.pine_ema_above_kama = Some(price_above_kama);
        self.pine_kama_long_filter = kama_rising;
        self.pine_kama_short_filter = kama_falling;

        self.pine_kama_slope_norm = if let Some(prev) = prev_kama.as_ref() {
            if atr.is_zero() {
                Some(Decimal::ZERO)
            } else {
                Some((kama - *prev) / atr)
            }
        } else {
            Some(Decimal::ZERO)
        };

        self.pine_kama_quality_score = if strong_buy {
            2
        } else if strong_sell {
            -2
        } else if buy_signal {
            1
        } else if sell_signal {
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
}
