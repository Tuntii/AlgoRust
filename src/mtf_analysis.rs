// =============================================================================
// MTF CONFLUENCE ANALYSIS MODULE
// Multi-Timeframe Confluence Scoring for Enhanced Signal Quality
// =============================================================================

use crate::types::Candle;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// =============================================================================
// ENUMS & TYPES
// =============================================================================

/// HTF Trend Bias (4h analysis)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendBias {
    StrongBullish, // EMA stack aligned bullish + HH/HL
    Bullish,       // Price > EMA200, general uptrend
    Neutral,       // No clear direction
    Bearish,       // Price < EMA200, general downtrend
    StrongBearish, // EMA stack aligned bearish + LH/LL
}

/// MTF Structure State (1h analysis)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructureState {
    ConfirmedBreak,    // BOS with displacement (+20)
    AtKeyLevel,        // Price at OB/demand zone (+15)
    PotentialReversal, // CHoCH detected (+10)
    Ranging,           // No clear structure (0)
    NoSetup,           // Nothing actionable (-5)
}

/// LTF Entry Quality (15m/30m analysis)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryQuality {
    Perfect,    // OB test + rejection + micro BOS (+15)
    Good,       // OB test OR FVG fill with confirmation (+10)
    Acceptable, // Basic setup (+5)
    Weak,       // Poor entry conditions (0)
}

/// EMA Stack Configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EMAStack {
    StrongBull, // EMA13 > EMA50 > EMA200 and price > all
    Bull,       // Price > EMA200
    Neutral,    // Mixed signals
    Bear,       // Price < EMA200
    StrongBear, // EMA13 < EMA50 < EMA200 and price < all
}

/// Swing Position in market structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwingPosition {
    HigherHighs, // Bullish structure
    LowerLows,   // Bearish structure
    Ranging,     // No clear structure
}

// =============================================================================
// ANALYSIS STRUCTS
// =============================================================================

/// HTF Analysis Result (4h timeframe)
#[derive(Debug, Clone)]
pub struct HTFAnalysis {
    pub ema_stack: EMAStack,
    pub ema200_slope: Decimal,
    pub swing_position: SwingPosition,
    pub major_support: Option<Decimal>,
    pub major_resistance: Option<Decimal>,
    pub bias: TrendBias,
}

/// MTF Analysis Result (1h timeframe)
#[derive(Debug, Clone)]
pub struct MTFAnalysis {
    pub has_recent_bos: bool,
    pub has_recent_choch: bool,
    pub has_displacement: bool,
    pub ob_proximity: Option<f64>, // 0.0 = at OB, 1.0 = far from OB
    pub structure_state: StructureState,
}

/// LTF Entry Analysis (15m/30m)
#[derive(Debug, Clone)]
pub struct LTFEntry {
    pub ob_test: bool,
    pub fvg_fill: bool,
    pub rejection_wick: bool,
    pub micro_bos: bool,
    pub momentum_shift: bool,
    pub quality: EntryQuality,
}

/// Final Confluence Result
#[derive(Debug, Clone, Serialize)]
pub struct ConfluenceResult {
    pub score: i32,
    pub htf_aligned: bool,
    pub mtf_aligned: bool,
    pub ltf_aligned: bool,
    pub alignment_count: u8,
    pub description: String,
}

// =============================================================================
// MTF CONFLUENCE ANALYZER
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct MTFConfluenceAnalyzer {
    /// Cached HTF analysis (updated less frequently)
    cached_htf: Option<HTFAnalysis>,
    last_htf_update: usize,

    /// Configuration
    htf_update_interval: usize, // How often to recompute HTF
}

impl MTFConfluenceAnalyzer {
    pub fn new() -> Self {
        Self {
            cached_htf: None,
            last_htf_update: 0,
            htf_update_interval: 4, // Update HTF every 4 candles
        }
    }

    /// Analyze HTF (Higher Timeframe - typically 4h)
    pub fn analyze_htf(
        &self,
        price: Decimal,
        ema13: Option<Decimal>,
        ema50: Option<Decimal>,
        ema200: Option<Decimal>,
        ema200_prev: Option<Decimal>,
        recent_highs: &[Decimal],
        recent_lows: &[Decimal],
    ) -> HTFAnalysis {
        // Determine EMA Stack
        let ema_stack = match (ema13, ema50, ema200) {
            (Some(e13), Some(e50), Some(e200)) => {
                if price > e13 && e13 > e50 && e50 > e200 {
                    EMAStack::StrongBull
                } else if price > e200 {
                    EMAStack::Bull
                } else if price < e13 && e13 < e50 && e50 < e200 {
                    EMAStack::StrongBear
                } else if price < e200 {
                    EMAStack::Bear
                } else {
                    EMAStack::Neutral
                }
            }
            (_, _, Some(e200)) => {
                if price > e200 {
                    EMAStack::Bull
                } else if price < e200 {
                    EMAStack::Bear
                } else {
                    EMAStack::Neutral
                }
            }
            _ => EMAStack::Neutral,
        };

        // Calculate EMA200 slope
        let ema200_slope = match (ema200, ema200_prev) {
            (Some(curr), Some(prev)) if prev != Decimal::ZERO => (curr - prev) / prev,
            _ => Decimal::ZERO,
        };

        // Determine Swing Position
        let swing_position = self.determine_swing_position(recent_highs, recent_lows);

        // Determine final bias
        let bias = self.calculate_htf_bias(&ema_stack, &swing_position, ema200_slope);

        HTFAnalysis {
            ema_stack,
            ema200_slope,
            swing_position,
            major_support: recent_lows.iter().min().cloned(),
            major_resistance: recent_highs.iter().max().cloned(),
            bias,
        }
    }

    /// Determine swing position from recent highs/lows
    fn determine_swing_position(&self, highs: &[Decimal], lows: &[Decimal]) -> SwingPosition {
        if highs.len() < 2 || lows.len() < 2 {
            return SwingPosition::Ranging;
        }

        // Check for Higher Highs and Higher Lows
        let hh = highs.windows(2).all(|w| w[1] >= w[0]);
        let hl = lows.windows(2).all(|w| w[1] >= w[0]);

        // Check for Lower Highs and Lower Lows
        let lh = highs.windows(2).all(|w| w[1] <= w[0]);
        let ll = lows.windows(2).all(|w| w[1] <= w[0]);

        if hh && hl {
            SwingPosition::HigherHighs
        } else if lh && ll {
            SwingPosition::LowerLows
        } else {
            SwingPosition::Ranging
        }
    }

    /// Calculate HTF bias from components
    fn calculate_htf_bias(
        &self,
        ema_stack: &EMAStack,
        swing_pos: &SwingPosition,
        slope: Decimal,
    ) -> TrendBias {
        let ema_score: i32 = match ema_stack {
            EMAStack::StrongBull => 3,
            EMAStack::Bull => 2,
            EMAStack::Neutral => 0,
            EMAStack::Bear => -2,
            EMAStack::StrongBear => -3,
        };

        let swing_score: i32 = match swing_pos {
            SwingPosition::HigherHighs => 2,
            SwingPosition::LowerLows => -2,
            SwingPosition::Ranging => 0,
        };

        let slope_score: i32 = if slope > Decimal::from_f64(0.001).unwrap() {
            1
        } else if slope < Decimal::from_f64(-0.001).unwrap() {
            -1
        } else {
            0
        };

        let total = ema_score + swing_score + slope_score;

        match total {
            5..=6 => TrendBias::StrongBullish,
            2..=4 => TrendBias::Bullish,
            -1..=1 => TrendBias::Neutral,
            -4..=-2 => TrendBias::Bearish,
            _ => TrendBias::StrongBearish,
        }
    }

    /// Analyze MTF (Mid Timeframe - typically 1h)
    pub fn analyze_mtf(
        &self,
        has_bos: bool,
        has_choch: bool,
        has_displacement: bool,
        ob_distance_pct: Option<f64>,
    ) -> MTFAnalysis {
        let structure_state = if has_bos && has_displacement {
            StructureState::ConfirmedBreak
        } else if ob_distance_pct.map(|d| d < 0.5).unwrap_or(false) {
            StructureState::AtKeyLevel
        } else if has_choch {
            StructureState::PotentialReversal
        } else if has_bos {
            StructureState::Ranging // BOS without confirmation
        } else {
            StructureState::NoSetup
        };

        MTFAnalysis {
            has_recent_bos: has_bos,
            has_recent_choch: has_choch,
            has_displacement,
            ob_proximity: ob_distance_pct,
            structure_state,
        }
    }

    /// Analyze LTF Entry (Lower Timeframe - typically 15m/30m)
    pub fn analyze_ltf_entry(
        &self,
        candle: &Candle,
        ob_tested: bool,
        fvg_filled: bool,
        has_micro_bos: bool,
        prev_momentum: Option<Decimal>,
        is_bullish_signal: bool,
    ) -> LTFEntry {
        let range = candle.high - candle.low;

        // Check for rejection wick
        let rejection_wick = if range > Decimal::ZERO {
            let body_top = candle.open.max(candle.close);
            let body_bottom = candle.open.min(candle.close);

            let (relevant_wick, wick_ratio) = if is_bullish_signal {
                // For bullish, we want lower wick (rejection of lower prices)
                let lower_wick = body_bottom - candle.low;
                (lower_wick, lower_wick / range)
            } else {
                // For bearish, we want upper wick
                let upper_wick = candle.high - body_top;
                (upper_wick, upper_wick / range)
            };

            wick_ratio > Decimal::from_f64(0.4).unwrap()
        } else {
            false
        };

        // Check momentum shift
        let momentum_shift = if let Some(prev) = prev_momentum {
            let current_momentum = candle.close - candle.open;
            if is_bullish_signal {
                prev < Decimal::ZERO && current_momentum > Decimal::ZERO
            } else {
                prev > Decimal::ZERO && current_momentum < Decimal::ZERO
            }
        } else {
            false
        };

        // Calculate quality score
        let mut score = 0;
        if ob_tested {
            score += 15;
        }
        if fvg_filled {
            score += 10;
        }
        if rejection_wick {
            score += 10;
        }
        if has_micro_bos {
            score += 15;
        }
        if momentum_shift {
            score += 10;
        }

        let quality = match score {
            40.. => EntryQuality::Perfect,
            25..=39 => EntryQuality::Good,
            10..=24 => EntryQuality::Acceptable,
            _ => EntryQuality::Weak,
        };

        LTFEntry {
            ob_test: ob_tested,
            fvg_fill: fvg_filled,
            rejection_wick,
            micro_bos: has_micro_bos,
            momentum_shift,
            quality,
        }
    }

    /// Calculate final MTF Confluence Score
    pub fn calculate_confluence(
        &self,
        htf: &HTFAnalysis,
        mtf: &MTFAnalysis,
        ltf: &LTFEntry,
        signal_direction_bullish: bool,
    ) -> ConfluenceResult {
        // Check HTF alignment
        let htf_aligned = match (signal_direction_bullish, htf.bias) {
            (true, TrendBias::StrongBullish | TrendBias::Bullish) => true,
            (false, TrendBias::StrongBearish | TrendBias::Bearish) => true,
            _ => false,
        };

        // Check MTF alignment
        let mtf_aligned = matches!(
            mtf.structure_state,
            StructureState::ConfirmedBreak | StructureState::AtKeyLevel
        );

        // Check LTF alignment
        let ltf_aligned = matches!(ltf.quality, EntryQuality::Perfect | EntryQuality::Good);

        // Count alignments
        let alignment_count = [htf_aligned, mtf_aligned, ltf_aligned]
            .iter()
            .filter(|&&x| x)
            .count() as u8;

        // Base score from alignment
        let base_score = match alignment_count {
            3 => 40,  // Perfect confluence
            2 => 25,  // Good confluence
            1 => 10,  // Weak
            0 => -20, // Conflict - AVOID
            _ => 0,
        };

        // Bonuses
        let mut bonus = 0;

        // Strong HTF trend bonus
        if matches!(
            htf.bias,
            TrendBias::StrongBullish | TrendBias::StrongBearish
        ) {
            bonus += 10;
        }

        // Perfect entry bonus
        if ltf.quality == EntryQuality::Perfect {
            bonus += 10;
        }

        // Structure confirmation bonus
        if mtf.structure_state == StructureState::ConfirmedBreak {
            bonus += 5;
        }

        let final_score = base_score + bonus;

        let description = format!(
            "MTF Confluence: {}/3 aligned (HTF:{} MTF:{} LTF:{}) Score:{}",
            alignment_count,
            if htf_aligned { "✓" } else { "✗" },
            if mtf_aligned { "✓" } else { "✗" },
            if ltf_aligned { "✓" } else { "✗" },
            final_score,
        );

        ConfluenceResult {
            score: final_score,
            htf_aligned,
            mtf_aligned,
            ltf_aligned,
            alignment_count,
            description,
        }
    }

    /// Quick evaluation for existing signals
    /// Uses available data from SymbolContext to compute confluence
    pub fn quick_evaluate(
        &self,
        price: Decimal,
        ema13: Option<Decimal>,
        ema50: Option<Decimal>,
        ema200: Option<Decimal>,
        has_bos: bool,
        has_displacement: bool,
        is_bullish: bool,
        candle: &Candle,
    ) -> ConfluenceResult {
        // Simplified HTF analysis using current TF data
        let htf = self.analyze_htf(
            price,
            ema13,
            ema50,
            ema200,
            None, // No previous EMA for slope
            &[],
            &[],
        );

        // MTF analysis
        let mtf = self.analyze_mtf(
            has_bos,
            false, // No CHoCH tracking in quick eval
            has_displacement,
            None, // No OB distance in quick eval
        );

        // LTF entry - basic check
        let ltf = self.analyze_ltf_entry(
            candle, false,   // No specific OB test
            false,   // No FVG tracking
            has_bos, // Use BOS as micro confirmation
            None, is_bullish,
        );

        self.calculate_confluence(&htf, &mtf, &ltf, is_bullish)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_test_candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            open_time: Utc::now(),
            open: Decimal::from_f64(open).unwrap(),
            high: Decimal::from_f64(high).unwrap(),
            low: Decimal::from_f64(low).unwrap(),
            close: Decimal::from_f64(close).unwrap(),
            volume: Decimal::from(1000),
        }
    }

    #[test]
    fn test_ema_stack_strong_bull() {
        let analyzer = MTFConfluenceAnalyzer::new();
        let htf = analyzer.analyze_htf(
            Decimal::from(100),
            Some(Decimal::from(95)),
            Some(Decimal::from(90)),
            Some(Decimal::from(80)),
            Some(Decimal::from(79)),
            &[],
            &[],
        );

        assert_eq!(htf.ema_stack, EMAStack::StrongBull);
        assert!(matches!(
            htf.bias,
            TrendBias::StrongBullish | TrendBias::Bullish
        ));
    }

    #[test]
    fn test_confluence_perfect() {
        let analyzer = MTFConfluenceAnalyzer::new();

        let htf = HTFAnalysis {
            ema_stack: EMAStack::StrongBull,
            ema200_slope: Decimal::from_f64(0.002).unwrap(),
            swing_position: SwingPosition::HigherHighs,
            major_support: None,
            major_resistance: None,
            bias: TrendBias::StrongBullish,
        };

        let mtf = MTFAnalysis {
            has_recent_bos: true,
            has_recent_choch: false,
            has_displacement: true,
            ob_proximity: Some(0.3),
            structure_state: StructureState::ConfirmedBreak,
        };

        let ltf = LTFEntry {
            ob_test: true,
            fvg_fill: true,
            rejection_wick: true,
            micro_bos: true,
            momentum_shift: true,
            quality: EntryQuality::Perfect,
        };

        let result = analyzer.calculate_confluence(&htf, &mtf, &ltf, true);

        assert_eq!(result.alignment_count, 3);
        assert!(result.score >= 40);
    }

    #[test]
    fn test_confluence_conflict() {
        let analyzer = MTFConfluenceAnalyzer::new();

        let htf = HTFAnalysis {
            ema_stack: EMAStack::StrongBear,
            ema200_slope: Decimal::from_f64(-0.002).unwrap(),
            swing_position: SwingPosition::LowerLows,
            major_support: None,
            major_resistance: None,
            bias: TrendBias::StrongBearish,
        };

        let mtf = MTFAnalysis {
            has_recent_bos: false,
            has_recent_choch: false,
            has_displacement: false,
            ob_proximity: None,
            structure_state: StructureState::NoSetup,
        };

        let ltf = LTFEntry {
            ob_test: false,
            fvg_fill: false,
            rejection_wick: false,
            micro_bos: false,
            momentum_shift: false,
            quality: EntryQuality::Weak,
        };

        // Trying to go LONG in strong bearish HTF
        let result = analyzer.calculate_confluence(&htf, &mtf, &ltf, true);

        assert_eq!(result.alignment_count, 0);
        assert!(result.score < 0);
    }
}
