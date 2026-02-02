use crate::types::{Candle, MarketStructure, TrendState, ContextId};
use crate::indicators::{Ema, Atr, is_pivot_high, is_pivot_low};
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
    pub atr_ratio_history: VecDeque<Decimal>, // Median ATR hesaplamak için tarihçe
    pub ema_50_slope_history: VecDeque<Decimal>, // EMA50 tarihçesi eğim hesabı için

    // Events
    pub just_confirmed_pivot_high: bool,
    pub just_confirmed_pivot_low: bool,
    pub last_signal_candle: Option<usize>, // Son sinyal üretilen mum indeksi (cooldown için)
    
    // BOS/Liquidity tracking
    pub just_broke_high: bool,  // Bu mumda high kırıldı mı?
    pub just_broke_low: bool,   // Bu mumda low kırıldı mı?
    pub pivot_high_history: VecDeque<Decimal>, // Son pivot high'lar (equal high tespiti için)
    pub pivot_low_history: VecDeque<Decimal>,  // Son pivot low'lar (equal low tespiti için)
    
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
            ema_5: Ema::new(5),
            ema_8: Ema::new(8),
            ema_13: Ema::new(13),
            ema_50: Ema::new(50),
            ema_200: Ema::new(200),
            atr_14: Atr::new(14),
            just_confirmed_pivot_high: false,
            just_confirmed_pivot_low: false,
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

        // Update EMAs
        let close = candle.close;
        self.ema_5.update(close);
        self.ema_8.update(close);
        self.ema_13.update(close);
        let cur_ema50 = self.ema_50.update(close);
        self.ema_200.update(close);

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

        // Store Candle - HTF needs more history
        self.candles.push_back(candle);
        let max_candles = BootstrapState::min_candles_for_tf(&self.timeframe).max(1500);
        if self.candles.len() > max_candles {
            self.candles.pop_front();
        }

        self.update_structure();
        
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
        if idx < 3 { return; } 

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        if is_pivot_high(&highs, idx) {
            let pivot_val = highs[idx];
            self.structure.last_pivot_high = Some(pivot_val);
            self.just_confirmed_pivot_high = true;
            self.last_pivot_high_idx = Some(self.total_candles_processed); // Track pivot index
            
            // Pivot history'e ekle (equal high tespiti için)
            self.pivot_high_history.push_back(pivot_val);
            if self.pivot_high_history.len() > 5 { self.pivot_high_history.pop_front(); }
            
            // Equal High kontrolü: Son 5 pivot high içinde %0.15 toleransla eşit var mı?
            self.structure.has_equal_highs = self.check_equal_levels(&self.pivot_high_history.clone());
        }
        
        if is_pivot_low(&lows, idx) {
            let pivot_val = lows[idx];
            self.structure.last_pivot_low = Some(pivot_val);
            self.just_confirmed_pivot_low = true;
            self.last_pivot_low_idx = Some(self.total_candles_processed); // Track pivot index
            
            // Pivot history'e ekle
            self.pivot_low_history.push_back(pivot_val);
            if self.pivot_low_history.len() > 5 { self.pivot_low_history.pop_front(); }
            
            // Equal Low kontrolü
            self.structure.has_equal_lows = self.check_equal_levels(&self.pivot_low_history.clone());
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

