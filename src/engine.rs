use crate::types::{TradeSignal, SignalType, TrendState};
use crate::state::SymbolContext;
use chrono::Utc;

pub struct SignalEngine;

impl SignalEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub fn evaluate(&self, ctx: &SymbolContext) -> Option<TradeSignal> {
        let mut reasons = Vec::new();
        let mut score = 0;
        let mut signal_type = None;

        let last_close = ctx.candles.back()?.close;

        match ctx.structure.trend {
            TrendState::Bullish => {
                if ctx.just_confirmed_pivot_low {
                     reasons.push("Bullish market structure confirmed".to_string());
                     reasons.push("Fractal HL detected (Pivot Low confirmed)".to_string());
                     reasons.push("EMA Alignment: 5>8>13>50>200".to_string());
                     
                     score += 30; // Structure
                     score += 25; // EMA
                     score += 20; // Pivot
                     
                     // Entry verification: Close > EMA50? (Already checked in TrendState, but explicit check good)
                     if let (Some(e50), Some(e13)) = (ctx.ema_50.current_value, ctx.ema_13.current_value) {
                         if last_close > e50 && last_close > e13 {
                             score += 10; // Context Confluence
                             reasons.push("Price above EMA13 & EMA50".to_string());
                         }
                     }
                     
                     signal_type = Some(SignalType::LONG);
                }
            },
            TrendState::Bearish => {
                if ctx.just_confirmed_pivot_high {
                     reasons.push("Bearish market structure confirmed".to_string());
                     reasons.push("Fractal LH detected (Pivot High confirmed)".to_string());
                     reasons.push("EMA Alignment: 5<8<13<50<200".to_string());
                     
                     score += 30; // Structure
                     score += 25; // EMA
                     score += 20; // Pivot
                     
                     if let (Some(e50), Some(e13)) = (ctx.ema_50.current_value, ctx.ema_13.current_value) {
                         if last_close < e50 && last_close < e13 {
                             score += 10;
                             reasons.push("Price below EMA13 & EMA50".to_string());
                         }
                     }

                     signal_type = Some(SignalType::SHORT);
                }
            },
            _ => {}
        }

        if let Some(sig) = signal_type {
            if score >= 70 {
                 return Some(TradeSignal {
                     symbol: ctx.symbol.clone(),
                     timeframe: ctx.timeframe.clone(),
                     signal: sig,
                     price: last_close,
                     confidence: score,
                     timestamp: Utc::now(),
                     reasons,
                 });
            }
        }
        
        None
    }
}
