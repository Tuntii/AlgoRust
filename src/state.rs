use crate::types::{Candle, MarketStructure, TrendState};
use crate::indicators::{Ema, is_pivot_high, is_pivot_low};
use std::collections::VecDeque;

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

    // Events
    pub just_confirmed_pivot_high: bool,
    pub just_confirmed_pivot_low: bool,
}

impl SymbolContext {
    pub fn new(symbol: String, timeframe: String) -> Self {
        Self {
            symbol,
            timeframe,
            candles: VecDeque::new(),
            structure: MarketStructure::default(),
            ema_5: Ema::new(5),
            ema_8: Ema::new(8),
            ema_13: Ema::new(13),
            ema_50: Ema::new(50),
            ema_200: Ema::new(200),
            just_confirmed_pivot_high: false,
            just_confirmed_pivot_low: false,
        }
    }

    pub fn add_candle(&mut self, candle: Candle) {
        // Reset events
        self.just_confirmed_pivot_high = false;
        self.just_confirmed_pivot_low = false;

        // Update EMAs
        let close = candle.close;
        self.ema_5.update(close);
        self.ema_8.update(close);
        self.ema_13.update(close);
        self.ema_50.update(close);
        self.ema_200.update(close);

        // Store Candle
        self.candles.push_back(candle);
        if self.candles.len() > 1500 {
            self.candles.pop_front();
        }

        self.update_structure();
    }

    fn update_structure(&mut self) {
        if self.candles.len() < 7 {
            return;
        }

        // Check Pivot at index len - 4
        // History: [..., P-3, P-2, P-1, P, P+1, P+2, P+3(current)]
        // Lookback 3, Lookahead 3.
        // Current index is len-1. We check pivot at len-1-3 = len-4?
        // Wait, lookahead 3 means we need 3 candles *after* the pivot.
        // If current is C (index i), and we check pivot at P (index i-3).
        // Then we have P+1 (i-2), P+2 (i-1), P+3 (i). Yes.
        
        let idx = self.candles.len().saturating_sub(4);
        if idx < 3 { return; } 

        let highs: Vec<_> = self.candles.iter().map(|c| c.high).collect();
        let lows: Vec<_> = self.candles.iter().map(|c| c.low).collect();

        if is_pivot_high(&highs, idx) {
            self.structure.last_pivot_high = Some(highs[idx]);
            self.just_confirmed_pivot_high = true;
        }
        
        if is_pivot_low(&lows, idx) {
            self.structure.last_pivot_low = Some(lows[idx]);
            self.just_confirmed_pivot_low = true;
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
}

