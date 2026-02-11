use crate::types::{Candle, MarketStructure, TrendState, ContextId};
use crate::indicators::{
    Adx, Atr, Ema, Kama, Macd, Rsi, StochRsi, check_divergence, is_pivot_high, is_pivot_low,
    DivergenceType,
};
use crate::policy::BootstrapState;
use std::collections::VecDeque;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;

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

    // Pine strategy indicators (15m optimized)
    pub kama_10: Kama,
    pub macd_8_21_5: Macd,
    pub adx_10_10: Adx,
    pub atr_10: Atr,
    pub rsi_10: Rsi,
    pub stoch_rsi_10: StochRsi,

    pub kama_10_series: VecDeque<Decimal>,
    pub rsi_10_series: VecDeque<Option<Decimal>>,
    pub stoch_k_series: VecDeque<Option<Decimal>>,

    pub pine_supertrend: Option<Decimal>,
    pub pine_trend: i32,
    pub pine_trend_changed_bullish: bool,
    pub pine_trend_changed_bearish: bool,
    pub pine_ema_cross_above: bool,
    pub pine_ema_cross_below: bool,
    pub pine_ema_above_kama: Option<bool>,
    pub pine_kama_long_filter: bool,
    pub pine_kama_short_filter: bool,
    pub pine_kama_quality_score: i32,
    pub pine_kama_slope_norm: Option<Decimal>,

    pub pine_bullish_div: bool,
    pub pine_bearish_div: bool,
    pub pine_rsi_bullish_div: bool,
    pub pine_rsi_bearish_div: bool,
    pub pine_stoch_bullish_div: bool,
    pub pine_stoch_bearish_div: bool,

    pub pine_last_price_high: Option<Decimal>,
    pub pine_last_price_low: Option<Decimal>,
    pub pine_last_price_high_bar: Option<usize>,
    pub pine_last_price_low_bar: Option<usize>,

    pub pine_last_rsi_high: Option<Decimal>,
    pub pine_last_rsi_low: Option<Decimal>,
    pub pine_last_rsi_high_bar: Option<usize>,
    pub pine_last_rsi_low_bar: Option<usize>,

    pub pine_last_stoch_high: Option<Decimal>,
    pub pine_last_stoch_low: Option<Decimal>,
    pub pine_last_stoch_high_bar: Option<usize>,
    pub pine_last_stoch_low_bar: Option<usize>,

    pub atr_ratio_history: VecDeque<Decimal>, // Median ATR hesaplamak için tarihçe
    pub ema_50_slope_history: VecDeque<Decimal>, // EMA50 tarihçesi eğim hesabı için
    pub rsi_history: VecDeque<(usize, Decimal)>, // RSI history (index, value)

    // Events
    pub just_confirmed_pivot_high: bool,
    pub just_confirmed_pivot_low: bool,
    pub last_signal_candle: Option<usize>, // Son sinyal üretilen mum indeksi (cooldown için)
    
    // BOS/Liquidity tracking
    pub just_broke_high: bool,  // Bu mumda high kırıldı mı?
    pub just_broke_low: bool,   // Bu mumda low kırıldı mı?
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
            kama_10: Kama::new(10, 2, 20),
            macd_8_21_5: Macd::new(8, 21, 5),
            adx_10_10: Adx::new(10, 10),
            atr_10: Atr::new(10),
            rsi_10: Rsi::new(10),
            stoch_rsi_10: StochRsi::new(10, 3, 3),
            kama_10_series: VecDeque::new(),
            rsi_10_series: VecDeque::new(),
            stoch_k_series: VecDeque::new(),
            pine_supertrend: None,
            pine_trend: 1,
            pine_trend_changed_bullish: false,
            pine_trend_changed_bearish: false,
            pine_ema_cross_above: false,
            pine_ema_cross_below: false,
            pine_ema_above_kama: None,
            pine_kama_long_filter: false,
            pine_kama_short_filter: false,
            pine_kama_quality_score: 0,
            pine_kama_slope_norm: None,
            pine_bullish_div: false,
            pine_bearish_div: false,
            pine_rsi_bullish_div: false,
            pine_rsi_bearish_div: false,
            pine_stoch_bullish_div: false,
            pine_stoch_bearish_div: false,
            pine_last_price_high: None,
            pine_last_price_low: None,
            pine_last_price_high_bar: None,
            pine_last_price_low_bar: None,
            pine_last_rsi_high: None,
            pine_last_rsi_low: None,
            pine_last_rsi_high_bar: None,
            pine_last_rsi_low_bar: None,
            pine_last_stoch_high: None,
            pine_last_stoch_low: None,
            pine_last_stoch_high_bar: None,
            pine_last_stoch_low_bar: None,
            just_confirmed_pivot_high: false,
            just_confirmed_pivot_low: false,
            current_divergence: DivergenceType::None,
            last_signal_candle: None,
            just_broke_high: false,
            just_broke_low: false,
            bootstrap: BootstrapState::with_timeframe(&timeframe), // TF-aware bootstrap
            total_candles_processed: 0,
            // Multi-position fields
            current_context_id: None,
            last_bos_candle_idx: None,
            last_pivot_high_idx: None,
            last_pivot_low_idx: None,
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
        if self.atr_ratio_history.is_empty() { return Decimal::ZERO; }
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
        if len < 7 { return Decimal::ZERO; }
        let current = self.ema_50_slope_history.back().unwrap();
        let old = self.ema_50_slope_history.get(len - 7).unwrap();
        
        if old.is_zero() { return Decimal::ZERO; }
        (*current - *old) / *old
    }
    
    // T5.2: Get average ATR for regime determination
    pub fn get_avg_atr(&self) -> Decimal {
        if let Some(atr) = self.atr_14.current_value {
            // Simple approximation using current ATR as baseline
            // In production, could track ATR history for better average
            atr
        } else {
            Decimal::ZERO
        }
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
        if self.kama_10_series.len() > 2000 { self.kama_10_series.pop_front(); }
        let _ = self.macd_8_21_5.update(close);
        let _ = self.adx_10_10.update(candle.high, candle.low, candle.close);
        let _ = self.atr_10.update(candle.high, candle.low, candle.close);
        let rsi_10_val = self.rsi_10.update(candle.close);
        self.rsi_10_series.push_back(rsi_10_val);
        if self.rsi_10_series.len() > 2000 { self.rsi_10_series.pop_front(); }

        let stoch_update = if let Some(rsi_val) = rsi_10_val {
            self.stoch_rsi_10.update(rsi_val)
        } else {
            None
        };
        let stoch_k_val = stoch_update.map(|(k, _)| k);
        self.stoch_k_series.push_back(stoch_k_val);
        if self.stoch_k_series.len() > 2000 { self.stoch_k_series.pop_front(); }

        // Update ATR
        if let Some(atr_val) = self.atr_14.update(candle.high, candle.low, candle.close) {
             if !candle.close.is_zero() {
                 let ratio = atr_val / candle.close;
                 self.atr_ratio_history.push_back(ratio);
                 if self.atr_ratio_history.len() > 200 { self.atr_ratio_history.pop_front(); }
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
        
        // Track EMA50 history - EMA update() returns Decimal directly
        self.ema_50_slope_history.push_back(cur_ema50);
        if self.ema_50_slope_history.len() > 20 { self.ema_50_slope_history.pop_front(); }

        // Update RSI
        if let Some(rsi_val) = self.rsi_14.update(candle.close) {
             self.rsi_history.push_back((self.total_candles_processed, rsi_val));
             // Keep history manageable (e.g., last 300)
             if self.rsi_history.len() > 300 { self.rsi_history.pop_front(); }
        }

        // Store Candle - HTF needs more history
        self.candles.push_back(candle);
        let max_candles = BootstrapState::min_candles_for_tf(&self.timeframe).max(1500);
        if self.candles.len() > max_candles {
            self.candles.pop_front();
        }

        self.update_structure();

        // Pine SuperTrend + EMAxKAMA cross + divergence
        self.update_pine_state(close, cur_kama);
        
        // T0.2 — Update Bootstrap State (TF-aware)
        let pivot_count = self.pivot_high_history.len().min(self.pivot_low_history.len());
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
        
        if idx < 3 { return; } 

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        if is_pivot_high(&highs, idx) {
            let pivot_val = highs[idx];
            self.structure.last_pivot_high = Some(pivot_val);
            self.just_confirmed_pivot_high = true;
            self.last_pivot_high_idx = Some(pivot_real_idx); // Track pivot index
            
            // Pivot history'e ekle (equal high tespiti için)
            self.pivot_high_history.push_back(pivot_val);
            if self.pivot_high_history.len() > 5 { self.pivot_high_history.pop_front(); }
            
            self.pivot_highs_with_idx.push((pivot_real_idx, pivot_val));
            if self.pivot_highs_with_idx.len() > 10 { self.pivot_highs_with_idx.remove(0); }

            // Equal High kontrolü: Son 5 pivot high içinde %0.15 toleransla eşit var mı?
            self.structure.has_equal_highs = self.check_equal_levels(&self.pivot_high_history.clone());

            // Check Bearish Divergence (HH price, LH RSI)
            let rsi_vec: Vec<_> = self.rsi_history.iter().cloned().collect();
            let divergence = check_divergence(&self.pivot_highs_with_idx, &self.pivot_lows_with_idx, &rsi_vec);
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
            if self.pivot_low_history.len() > 5 { self.pivot_low_history.pop_front(); }
            
            self.pivot_lows_with_idx.push((pivot_real_idx, pivot_val));
            if self.pivot_lows_with_idx.len() > 10 { self.pivot_lows_with_idx.remove(0); }

            // Equal Low kontrolü
            self.structure.has_equal_lows = self.check_equal_levels(&self.pivot_low_history.clone());

            // Check Bullish Divergence (LL price, HL RSI)
            let rsi_vec: Vec<_> = self.rsi_history.iter().cloned().collect();
            let divergence = check_divergence(&self.pivot_highs_with_idx, &self.pivot_lows_with_idx, &rsi_vec);
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
        let ema = match self.ema_13.current_value {
            Some(val) => val,
            None => return,
        };
        let atr = match self.atr_10.current_value {
            Some(val) => val,
            None => return,
        };
        let macd_hist = match self.macd_8_21_5.hist {
            Some(val) => val,
            None => return,
        };
        let adx = match self.adx_10_10.adx {
            Some(val) => val,
            None => return,
        };

        let macd_weight_k = Decimal::from_f64(0.5).unwrap();
        let weight_min = Decimal::from_f64(0.3).unwrap();
        let weight_max = Decimal::from_f64(0.7).unwrap();
        let adx_threshold = Decimal::from_f64(18.0).unwrap();
        let adx_weight_k = Decimal::from_f64(0.5).unwrap();
        let atr_multiplier = Decimal::from_f64(2.0).unwrap();

        let macd_hist_norm = if close.is_zero() { Decimal::ZERO } else { macd_hist / close };
        let ema_weight_raw = Decimal::from_f64(0.5).unwrap() + macd_weight_k * macd_hist_norm;
        let mut ema_weight = ema_weight_raw.clamp(weight_min, weight_max);

        let adx_factor = if adx > adx_threshold {
            Decimal::ONE + adx_weight_k * ((adx - adx_threshold) / adx_threshold)
        } else {
            Decimal::ONE
        };
        ema_weight = (ema_weight * adx_factor).clamp(weight_min, weight_max);

        let hybrid_base = ema * ema_weight + kama * (Decimal::ONE - ema_weight);
        let st_upper = hybrid_base + atr * atr_multiplier;
        let st_lower = hybrid_base - atr * atr_multiplier;

        let prev_trend = self.pine_trend;
        if self.pine_supertrend.is_none() {
            self.pine_trend = if close > hybrid_base { 1 } else { -1 };
            self.pine_supertrend = Some(if self.pine_trend == 1 { st_lower } else { st_upper });
        } else if self.pine_trend == 1 {
            if close < st_lower {
                self.pine_trend = -1;
                self.pine_supertrend = Some(st_upper);
            } else {
                let prev = self.pine_supertrend.unwrap_or(st_lower);
                self.pine_supertrend = Some(if st_lower > prev { st_lower } else { prev });
            }
        } else if close > st_upper {
            self.pine_trend = 1;
            self.pine_supertrend = Some(st_lower);
        } else {
            let prev = self.pine_supertrend.unwrap_or(st_upper);
            self.pine_supertrend = Some(if st_upper < prev { st_upper } else { prev });
        }

        self.pine_trend_changed_bullish = self.pine_trend == 1 && prev_trend == -1;
        self.pine_trend_changed_bearish = self.pine_trend == -1 && prev_trend == 1;

        let ema_above_kama = ema > kama;
        if let Some(prev) = self.pine_ema_above_kama {
            self.pine_ema_cross_above = !prev && ema_above_kama;
            self.pine_ema_cross_below = prev && !ema_above_kama;
        }
        self.pine_ema_above_kama = Some(ema_above_kama);

        // KAMA Quality Filter (15m-optimized):
        // 1) Trend alignment, 2) KAMA slope strength, 3) ADX trend strength,
        // 4) Reasonable distance from KAMA (avoid overextended entries),
        // 5) RSI regime, 6) Momentum confirmation.
        let slope_lookback = 4usize;
        let kama_prev = if self.kama_10_series.len() > slope_lookback {
            let idx = self.kama_10_series.len() - 1 - slope_lookback;
            self.kama_10_series.get(idx).copied().unwrap_or(kama)
        } else {
            kama
        };
        let slope_norm = if atr.is_zero() {
            Decimal::ZERO
        } else {
            (kama - kama_prev) / atr
        };
        self.pine_kama_slope_norm = Some(slope_norm);

        let dist_from_kama = if atr.is_zero() {
            Decimal::ZERO
        } else {
            (close - kama) / atr
        };

        let slope_threshold = Decimal::from_f64(0.12).unwrap();
        let adx_threshold = Decimal::from_f64(18.0).unwrap();
        let max_extension = Decimal::from_f64(1.8).unwrap();
        let long_pullback_min = Decimal::from_f64(-0.6).unwrap();
        let short_pullback_max = Decimal::from_f64(0.6).unwrap();

        let long_rsi_min = Decimal::from_f64(46.0).unwrap();
        let long_rsi_max = Decimal::from_f64(70.0).unwrap();
        let short_rsi_min = Decimal::from_f64(30.0).unwrap();
        let short_rsi_max = Decimal::from_f64(54.0).unwrap();

        let trend_long = close > kama && ema > kama && self.pine_trend == 1;
        let trend_short = close < kama && ema < kama && self.pine_trend == -1;
        let slope_long = slope_norm >= slope_threshold;
        let slope_short = slope_norm <= -slope_threshold;
        let adx_ok = adx >= adx_threshold;
        let long_distance_ok = dist_from_kama >= long_pullback_min && dist_from_kama <= max_extension;
        let short_distance_ok = dist_from_kama <= short_pullback_max && dist_from_kama >= -max_extension;
        let long_rsi_ok = self
            .rsi_10
            .current_value
            .map(|v| v >= long_rsi_min && v <= long_rsi_max)
            .unwrap_or(false);
        let short_rsi_ok = self
            .rsi_10
            .current_value
            .map(|v| v >= short_rsi_min && v <= short_rsi_max)
            .unwrap_or(false);
        let long_momentum_ok = macd_hist >= Decimal::ZERO;
        let short_momentum_ok = macd_hist <= Decimal::ZERO;

        let long_score = (if trend_long { 1 } else { 0 })
            + (if slope_long { 1 } else { 0 })
            + (if adx_ok { 1 } else { 0 })
            + (if long_distance_ok { 1 } else { 0 })
            + (if long_rsi_ok { 1 } else { 0 })
            + (if long_momentum_ok { 1 } else { 0 });
        let short_score = (if trend_short { 1 } else { 0 })
            + (if slope_short { 1 } else { 0 })
            + (if adx_ok { 1 } else { 0 })
            + (if short_distance_ok { 1 } else { 0 })
            + (if short_rsi_ok { 1 } else { 0 })
            + (if short_momentum_ok { 1 } else { 0 });

        self.pine_kama_long_filter = trend_long && slope_long && long_score >= 4;
        self.pine_kama_short_filter = trend_short && slope_short && short_score >= 4;
        self.pine_kama_quality_score = if self.pine_kama_long_filter {
            long_score
        } else if self.pine_kama_short_filter {
            -short_score
        } else {
            long_score - short_score
        };

        let lookback_left = 5usize;
        let lookback_right = 5usize;
        let max_div_range = 30usize;
        let rsi_div_min_pct = Decimal::from_f64(4.0).unwrap();
        let stoch_div_min_pct = Decimal::from_f64(4.0).unwrap();

        if self.candles.len() <= lookback_left + lookback_right {
            return;
        }

        let pivot_idx = self.candles.len() - 1 - lookback_right;
        if pivot_idx < lookback_left {
            return;
        }

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        let price_pivot_high = Self::is_pivot_high_lr(&highs, pivot_idx, lookback_left, lookback_right);
        let price_pivot_low = Self::is_pivot_low_lr(&lows, pivot_idx, lookback_left, lookback_right);

        let rsi_pivot_high = Self::is_pivot_high_opt(&self.rsi_10_series, pivot_idx, lookback_left, lookback_right);
        let rsi_pivot_low = Self::is_pivot_low_opt(&self.rsi_10_series, pivot_idx, lookback_left, lookback_right);

        let stoch_pivot_high = Self::is_pivot_high_opt(&self.stoch_k_series, pivot_idx, lookback_left, lookback_right);
        let stoch_pivot_low = Self::is_pivot_low_opt(&self.stoch_k_series, pivot_idx, lookback_left, lookback_right);

        let pivot_bar = self.total_candles_processed.saturating_sub(lookback_right);

        let prev_price_high = self.pine_last_price_high;
        let prev_price_low = self.pine_last_price_low;
        let prev_price_high_bar = self.pine_last_price_high_bar;
        let prev_price_low_bar = self.pine_last_price_low_bar;

        let prev_rsi_high = self.pine_last_rsi_high;
        let prev_rsi_low = self.pine_last_rsi_low;
        let prev_stoch_high = self.pine_last_stoch_high;
        let prev_stoch_low = self.pine_last_stoch_low;

        if price_pivot_high {
            self.pine_last_price_high = Some(highs[pivot_idx]);
            self.pine_last_price_high_bar = Some(pivot_bar);
        }
        if price_pivot_low {
            self.pine_last_price_low = Some(lows[pivot_idx]);
            self.pine_last_price_low_bar = Some(pivot_bar);
        }

        if rsi_pivot_high {
            if let Some(val) = Self::series_value(&self.rsi_10_series, pivot_idx) {
                self.pine_last_rsi_high = Some(val);
                self.pine_last_rsi_high_bar = Some(pivot_bar);
            }
        }
        if rsi_pivot_low {
            if let Some(val) = Self::series_value(&self.rsi_10_series, pivot_idx) {
                self.pine_last_rsi_low = Some(val);
                self.pine_last_rsi_low_bar = Some(pivot_bar);
            }
        }

        if stoch_pivot_high {
            if let Some(val) = Self::series_value(&self.stoch_k_series, pivot_idx) {
                self.pine_last_stoch_high = Some(val);
                self.pine_last_stoch_high_bar = Some(pivot_bar);
            }
        }
        if stoch_pivot_low {
            if let Some(val) = Self::series_value(&self.stoch_k_series, pivot_idx) {
                self.pine_last_stoch_low = Some(val);
                self.pine_last_stoch_low_bar = Some(pivot_bar);
            }
        }

        if price_pivot_low {
            if let (Some(prev_price), Some(prev_bar), Some(prev_rsi)) = (prev_price_low, prev_price_low_bar, prev_rsi_low) {
                let bars_since = pivot_bar.saturating_sub(prev_bar);
                if bars_since > 0 && bars_since <= max_div_range {
                    let price_ll = lows[pivot_idx] < prev_price;
                    let rsi_hl = if let Some(curr_rsi) = Self::series_value(&self.rsi_10_series, pivot_idx) {
                        curr_rsi > prev_rsi
                    } else {
                        false
                    };
                    if prev_rsi != Decimal::ZERO {
                        let rsi_diff_pct = (Self::series_value(&self.rsi_10_series, pivot_idx).unwrap_or(prev_rsi) - prev_rsi)
                            / prev_rsi * Decimal::from(100);
                        self.pine_rsi_bullish_div = price_ll && rsi_hl && rsi_diff_pct > rsi_div_min_pct;
                    }
                }
            }

            if let (Some(prev_price), Some(prev_bar), Some(prev_stoch)) = (prev_price_low, prev_price_low_bar, prev_stoch_low) {
                let bars_since = pivot_bar.saturating_sub(prev_bar);
                if bars_since > 0 && bars_since <= max_div_range {
                    let price_ll = lows[pivot_idx] < prev_price;
                    let stoch_hl = if let Some(curr_stoch) = Self::series_value(&self.stoch_k_series, pivot_idx) {
                        curr_stoch > prev_stoch
                    } else {
                        false
                    };
                    let denom = if prev_stoch.is_zero() { Decimal::from_f64(0.01).unwrap() } else { prev_stoch };
                    let stoch_diff_pct = (Self::series_value(&self.stoch_k_series, pivot_idx).unwrap_or(prev_stoch) - prev_stoch)
                        / denom * Decimal::from(100);
                    self.pine_stoch_bullish_div = price_ll && stoch_hl && stoch_diff_pct > stoch_div_min_pct;
                }
            }
        }

        if price_pivot_high {
            if let (Some(prev_price), Some(prev_bar), Some(prev_rsi)) = (prev_price_high, prev_price_high_bar, prev_rsi_high) {
                let bars_since = pivot_bar.saturating_sub(prev_bar);
                if bars_since > 0 && bars_since <= max_div_range {
                    let price_hh = highs[pivot_idx] > prev_price;
                    let rsi_lh = if let Some(curr_rsi) = Self::series_value(&self.rsi_10_series, pivot_idx) {
                        curr_rsi < prev_rsi
                    } else {
                        false
                    };
                    if prev_rsi != Decimal::ZERO {
                        let rsi_diff_pct = (prev_rsi - Self::series_value(&self.rsi_10_series, pivot_idx).unwrap_or(prev_rsi))
                            / prev_rsi * Decimal::from(100);
                        self.pine_rsi_bearish_div = price_hh && rsi_lh && rsi_diff_pct > rsi_div_min_pct;
                    }
                }
            }

            if let (Some(prev_price), Some(prev_bar), Some(prev_stoch)) = (prev_price_high, prev_price_high_bar, prev_stoch_high) {
                let bars_since = pivot_bar.saturating_sub(prev_bar);
                if bars_since > 0 && bars_since <= max_div_range {
                    let price_hh = highs[pivot_idx] > prev_price;
                    let stoch_lh = if let Some(curr_stoch) = Self::series_value(&self.stoch_k_series, pivot_idx) {
                        curr_stoch < prev_stoch
                    } else {
                        false
                    };
                    let denom = if prev_stoch.is_zero() { Decimal::from_f64(0.01).unwrap() } else { prev_stoch };
                    let stoch_diff_pct = (prev_stoch - Self::series_value(&self.stoch_k_series, pivot_idx).unwrap_or(prev_stoch))
                        / denom * Decimal::from(100);
                    self.pine_stoch_bearish_div = price_hh && stoch_lh && stoch_diff_pct > stoch_div_min_pct;
                }
            }
        }

        self.pine_bullish_div = self.pine_rsi_bullish_div || self.pine_stoch_bullish_div;
        self.pine_bearish_div = self.pine_rsi_bearish_div || self.pine_stoch_bearish_div;
    }

    fn is_pivot_high_lr(values: &[Decimal], idx: usize, left: usize, right: usize) -> bool {
        if idx < left || idx + right >= values.len() {
            return false;
        }
        let current = values[idx];
        values[idx - left..idx].iter().all(|&v| v < current)
            && values[idx + 1..=idx + right].iter().all(|&v| v < current)
    }

    fn is_pivot_low_lr(values: &[Decimal], idx: usize, left: usize, right: usize) -> bool {
        if idx < left || idx + right >= values.len() {
            return false;
        }
        let current = values[idx];
        values[idx - left..idx].iter().all(|&v| v > current)
            && values[idx + 1..=idx + right].iter().all(|&v| v > current)
    }

    fn is_pivot_high_opt(
        values: &VecDeque<Option<Decimal>>,
        idx: usize,
        left: usize,
        right: usize,
    ) -> bool {
        if idx < left || idx + right >= values.len() {
            return false;
        }
        let current = match values.get(idx).and_then(|v| *v) {
            Some(val) => val,
            None => return false,
        };
        for i in idx - left..=idx + right {
            if i == idx {
                continue;
            }
            if let Some(val) = values.get(i).and_then(|v| *v) {
                if val >= current {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn is_pivot_low_opt(
        values: &VecDeque<Option<Decimal>>,
        idx: usize,
        left: usize,
        right: usize,
    ) -> bool {
        if idx < left || idx + right >= values.len() {
            return false;
        }
        let current = match values.get(idx).and_then(|v| *v) {
            Some(val) => val,
            None => return false,
        };
        for i in idx - left..=idx + right {
            if i == idx {
                continue;
            }
            if let Some(val) = values.get(i).and_then(|v| *v) {
                if val <= current {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn series_value(values: &VecDeque<Option<Decimal>>, idx: usize) -> Option<Decimal> {
        values.get(idx).and_then(|v| *v)
    }
    
    /// Equal high/low tespiti: Son pivot'lar arasında %0.15 toleransla eşit seviye var mı?
    fn check_equal_levels(&self, pivots: &VecDeque<Decimal>) -> bool {
        if pivots.len() < 2 { return false; }
        
        let tolerance = Decimal::from_f64(0.0015).unwrap(); // %0.15
        
        for i in 0..pivots.len() {
            for j in (i+1)..pivots.len() {
                let p1 = pivots[i];
                let p2 = pivots[j];
                if p1.is_zero() { continue; }
                
                let diff_pct = ((p1 - p2) / p1).abs();
                if diff_pct < tolerance {
                    return true; // Equal level bulundu = Liquidity pool
                }
            }
        }
        false
    }
}

