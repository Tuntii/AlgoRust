// ============================================================================
// ANALYTICS MODULE — Phase 5 Backtest Analytics + Phase 6 Production Metrics
// ============================================================================

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::types::{RegimeContext, TradeSignal, SignalType};

// =============================================================================
// BLOCK STATISTICS TRACKER
// =============================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct BlockStats {
    pub wick_trap_blocks: u32,
    pub flat_ema_blocks: u32,
    pub low_atr_blocks: u32,
    pub bootstrap_incomplete: u32,
    pub cooldown_blocks: u32,
    pub open_trade_blocks: u32,  // Blocked because trade already open (LEGACY - single position)
    pub score_too_low: u32,
    pub policy_blocked: u32,
    pub total_evaluations: u32,
    pub total_signals_generated: u32,
    
    // Multi-Position Block Stats (TASK 2)
    pub max_trades_reached: u32,      // Blocked because max active trades reached
    pub duplicate_context: u32,       // Blocked because same context_id already active
    pub hedge_blocked: u32,           // Blocked because opposite direction trade exists
    pub context_cooldown_blocks: u32, // Blocked due to context-specific cooldown
    
    // Phase 8 Block Stats
    pub trend_saturation_blocks: u32, // T8.3: Blocked due to weakening trend
    pub weak_trade_replaced: u32,     // T8.2: Count of weak trades replaced
    
    // Phase 9 Stats
    pub max_duration_exits: u32,      // T9.1: Trades forced closed due to max duration
    pub be_applied_count: u32,        // T9.2: Trades where BE was applied
    pub partial_tp_count: u32,        // T9.3: Partial TPs taken
    
    // Phase 10 Stats
    pub kill_switch_triggered: u32,   // T10.2: Times kill switch was triggered
}

impl BlockStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn merge(&mut self, other: &BlockStats) {
        self.wick_trap_blocks += other.wick_trap_blocks;
        self.flat_ema_blocks += other.flat_ema_blocks;
        self.low_atr_blocks += other.low_atr_blocks;
        self.bootstrap_incomplete += other.bootstrap_incomplete;
        self.cooldown_blocks += other.cooldown_blocks;
        self.open_trade_blocks += other.open_trade_blocks;
        self.score_too_low += other.score_too_low;
        self.policy_blocked += other.policy_blocked;
        self.total_evaluations += other.total_evaluations;
        self.total_signals_generated += other.total_signals_generated;
        // Multi-position blocks
        self.max_trades_reached += other.max_trades_reached;
        self.duplicate_context += other.duplicate_context;
        self.hedge_blocked += other.hedge_blocked;
        self.context_cooldown_blocks += other.context_cooldown_blocks;
        // Phase 8 blocks
        self.trend_saturation_blocks += other.trend_saturation_blocks;
        self.weak_trade_replaced += other.weak_trade_replaced;
        // Phase 9 stats
        self.max_duration_exits += other.max_duration_exits;
        self.be_applied_count += other.be_applied_count;
        self.partial_tp_count += other.partial_tp_count;
        // Phase 10 stats
        self.kill_switch_triggered += other.kill_switch_triggered;
    }
    
    pub fn total_blocks(&self) -> u32 {
        self.wick_trap_blocks + 
        self.flat_ema_blocks + 
        self.low_atr_blocks + 
        self.bootstrap_incomplete + 
        self.cooldown_blocks + 
        self.open_trade_blocks +
        self.score_too_low +
        self.policy_blocked +
        // Multi-position blocks
        self.max_trades_reached +
        self.duplicate_context +
        self.hedge_blocked +
        self.context_cooldown_blocks +
        // Phase 8 blocks
        self.trend_saturation_blocks
    }
    
    pub fn signal_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.total_signals_generated as f64 / self.total_evaluations as f64 * 100.0
        }
    }
}

// =============================================================================
// T5.1 — Advanced Backtest Metrics
// =============================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct AdvancedMetrics {
    pub expectancy_r: f64,           // Expected R per trade
    pub max_consecutive_losses: u32,
    pub max_consecutive_wins: u32,
    pub avg_r_per_trade: f64,
    pub avg_win_r: f64,
    pub avg_loss_r: f64,
    pub profit_factor: f64,          // gross_wins / gross_losses
    pub largest_win_r: f64,
    pub largest_loss_r: f64,
    pub avg_trade_duration_candles: f64,
    pub sharpe_ratio_approx: f64,    // Simplified Sharpe
    pub trade_count: u32,
    
    // Multi-Position Metrics (TASK 6)
    pub max_concurrent_trades: u32,
    pub avg_concurrent_trades: f64,
    pub overlap_trade_count: u32,     // Trades that overlapped with another
    pub overlap_win_rate: f64,        // Win rate of overlapping trades
    pub overlap_pnl_r: f64,           // Total PnL from overlapping trades
    pub context_based_expectancy: HashMap<String, f64>, // Expectancy by context type
    
    // T10.1: Overlap Risk Metrics
    pub pnl_by_overlap_count: HashMap<u32, f64>, // PnL grouped by how many concurrent trades
    pub worst_overlap_drawdown: f64,             // Worst drawdown during overlap periods
    pub overlap_1_pnl: f64,                      // PnL when 1 trade active
    pub overlap_2_pnl: f64,                      // PnL when 2 trades active
    pub overlap_3_pnl: f64,                      // PnL when 3+ trades active
}

#[derive(Debug, Clone, Serialize)]
pub struct TradeRecord {
    pub pnl_r: f64,
    pub is_win: bool,
    pub duration_candles: u32,
    pub regime: Option<RegimeContext>,
    pub confidence_tier: String,
    
    // Multi-Position fields (TASK 6)
    pub context_type: Option<String>,      // "bos", "liquidity_sweep", "pivot"
    pub opened_at_candle: Option<usize>,   // For overlap detection
    pub exit_candle_idx: Option<usize>,    // For overlap detection
    pub adjusted_confidence: Option<u8>,   // Confidence after multi-trade adjustment
    pub was_concurrent: bool,              // Was this trade concurrent with another?
}

impl AdvancedMetrics {
    pub fn calculate(trades: &[TradeRecord]) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let trade_count = trades.len() as u32;
        
        // Basic stats
        let total_r: f64 = trades.iter().map(|t| t.pnl_r).sum();
        let wins: Vec<&TradeRecord> = trades.iter().filter(|t| t.is_win).collect();
        let losses: Vec<&TradeRecord> = trades.iter().filter(|t| !t.is_win).collect();
        
        let win_count = wins.len() as f64;
        let loss_count = losses.len() as f64;
        
        // Win/Loss R averages
        let avg_win_r = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl_r).sum::<f64>() / win_count
        } else { 0.0 };
        
        let avg_loss_r = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl_r.abs()).sum::<f64>() / loss_count
        } else { 0.0 };
        
        // Expectancy: (Win% × AvgWin) - (Loss% × AvgLoss)
        let win_rate = win_count / (trade_count as f64);
        let loss_rate = 1.0 - win_rate;
        let expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r);
        
        // Profit Factor
        let gross_wins: f64 = wins.iter().map(|t| t.pnl_r).sum();
        let gross_losses: f64 = losses.iter().map(|t| t.pnl_r.abs()).sum();
        let profit_factor = if gross_losses > 0.0 {
            gross_wins / gross_losses
        } else if gross_wins > 0.0 { 
            f64::INFINITY 
        } else { 
            0.0 
        };
        
        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = Self::calc_consecutive(trades);
        
        // Largest win/loss
        let largest_win_r = trades.iter().map(|t| t.pnl_r).fold(0.0_f64, f64::max);
        let largest_loss_r = trades.iter().map(|t| t.pnl_r).fold(0.0_f64, f64::min).abs();
        
        // Average trade duration
        let avg_trade_duration_candles = trades.iter()
            .map(|t| t.duration_candles as f64)
            .sum::<f64>() / trade_count as f64;
        
        // Simplified Sharpe Ratio (mean / std_dev of R returns)
        let avg_r = total_r / trade_count as f64;
        let variance = trades.iter()
            .map(|t| (t.pnl_r - avg_r).powi(2))
            .sum::<f64>() / trade_count as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio_approx = if std_dev > 0.0 { avg_r / std_dev } else { 0.0 };
        
        // Multi-Position Metrics (TASK 6)
        let (max_concurrent, avg_concurrent, overlap_count, overlap_wins, overlap_pnl) = 
            Self::calc_concurrent_metrics(trades);
        
        let overlap_win_rate = if overlap_count > 0 {
            overlap_wins as f64 / overlap_count as f64 * 100.0
        } else {
            0.0
        };
        
        // Context-based expectancy
        let context_based_expectancy = Self::calc_context_expectancy(trades);
        
        // T10.1: Overlap Risk Metrics
        let (pnl_by_overlap, worst_drawdown, overlap_1, overlap_2, overlap_3) = 
            Self::calc_overlap_risk_metrics(trades);
        
        Self {
            expectancy_r,
            max_consecutive_losses,
            max_consecutive_wins,
            avg_r_per_trade: avg_r,
            avg_win_r,
            avg_loss_r,
            profit_factor,
            largest_win_r,
            largest_loss_r,
            avg_trade_duration_candles,
            sharpe_ratio_approx,
            trade_count,
            max_concurrent_trades: max_concurrent,
            avg_concurrent_trades: avg_concurrent,
            overlap_trade_count: overlap_count,
            overlap_win_rate,
            overlap_pnl_r: overlap_pnl,
            context_based_expectancy,
            // T10.1: Overlap Risk
            pnl_by_overlap_count: pnl_by_overlap,
            worst_overlap_drawdown: worst_drawdown,
            overlap_1_pnl: overlap_1,
            overlap_2_pnl: overlap_2,
            overlap_3_pnl: overlap_3,
        }
    }
    
    fn calc_consecutive(trades: &[TradeRecord]) -> (u32, u32) {
        let mut max_wins = 0u32;
        let mut max_losses = 0u32;
        let mut cur_wins = 0u32;
        let mut cur_losses = 0u32;
        
        for trade in trades {
            if trade.is_win {
                cur_wins += 1;
                cur_losses = 0;
                max_wins = max_wins.max(cur_wins);
            } else {
                cur_losses += 1;
                cur_wins = 0;
                max_losses = max_losses.max(cur_losses);
            }
        }
        
        (max_wins, max_losses)
    }
    
    /// Calculate concurrent trade metrics
    fn calc_concurrent_metrics(trades: &[TradeRecord]) -> (u32, f64, u32, u32, f64) {
        if trades.is_empty() {
            return (0, 0.0, 0, 0, 0.0);
        }
        
        let mut max_concurrent: u32 = 1;
        let mut overlap_count: u32 = 0;
        let mut overlap_wins: u32 = 0;
        let mut overlap_pnl: f64 = 0.0;
        let mut total_concurrent: f64 = 0.0;
        
        for (i, trade) in trades.iter().enumerate() {
            let t_start = trade.opened_at_candle.unwrap_or(0);
            let t_end = trade.exit_candle_idx.unwrap_or(t_start);
            
            let mut concurrent = 1u32;
            let mut has_overlap = false;
            
            for (j, other) in trades.iter().enumerate() {
                if i == j { continue; }
                
                let o_start = other.opened_at_candle.unwrap_or(0);
                let o_end = other.exit_candle_idx.unwrap_or(o_start);
                
                // Check for overlap
                if o_start <= t_end && o_end >= t_start {
                    concurrent += 1;
                    has_overlap = true;
                }
            }
            
            max_concurrent = max_concurrent.max(concurrent);
            total_concurrent += concurrent as f64;
            
            // Track overlap metrics
            if has_overlap || trade.was_concurrent {
                overlap_count += 1;
                overlap_pnl += trade.pnl_r;
                if trade.is_win {
                    overlap_wins += 1;
                }
            }
        }
        
        let avg_concurrent = total_concurrent / trades.len() as f64;
        
        (max_concurrent, avg_concurrent, overlap_count, overlap_wins, overlap_pnl)
    }
    
    /// Calculate expectancy by context type
    fn calc_context_expectancy(trades: &[TradeRecord]) -> HashMap<String, f64> {
        let mut by_context: HashMap<String, Vec<f64>> = HashMap::new();
        
        for trade in trades {
            let ctx = trade.context_type.clone().unwrap_or_else(|| "unknown".to_string());
            by_context.entry(ctx).or_default().push(trade.pnl_r);
        }
        
        let mut result = HashMap::new();
        for (ctx, pnls) in by_context {
            if !pnls.is_empty() {
                let expectancy = pnls.iter().sum::<f64>() / pnls.len() as f64;
                result.insert(ctx, expectancy);
            }
        }
        
        result
    }
    
    /// T10.1: Calculate overlap risk metrics
    fn calc_overlap_risk_metrics(trades: &[TradeRecord]) -> (HashMap<u32, f64>, f64, f64, f64, f64) {
        let mut pnl_by_overlap: HashMap<u32, f64> = HashMap::new();
        let mut overlap_1_pnl = 0.0;
        let mut overlap_2_pnl = 0.0;
        let mut overlap_3_pnl = 0.0;
        
        // Calculate concurrent count for each trade
        for (i, trade) in trades.iter().enumerate() {
            let t_start = trade.opened_at_candle.unwrap_or(0);
            let t_end = trade.exit_candle_idx.unwrap_or(t_start);
            
            let mut concurrent_count = 1u32;
            
            for (j, other) in trades.iter().enumerate() {
                if i == j { continue; }
                
                let o_start = other.opened_at_candle.unwrap_or(0);
                let o_end = other.exit_candle_idx.unwrap_or(o_start);
                
                // Check for overlap
                if o_start <= t_end && o_end >= t_start {
                    concurrent_count += 1;
                }
            }
            
            // Add PnL to the appropriate bucket
            *pnl_by_overlap.entry(concurrent_count).or_insert(0.0) += trade.pnl_r;
            
            match concurrent_count {
                1 => overlap_1_pnl += trade.pnl_r,
                2 => overlap_2_pnl += trade.pnl_r,
                _ => overlap_3_pnl += trade.pnl_r,
            }
        }
        
        // Calculate worst overlap drawdown (worst consecutive losing streak during overlap)
        let mut worst_overlap_drawdown: f64 = 0.0;
        let mut current_drawdown: f64 = 0.0;
        
        // Filter to overlapping trades only
        let overlap_trades: Vec<&TradeRecord> = trades.iter()
            .filter(|t| t.was_concurrent)
            .collect();
        
        for trade in &overlap_trades {
            if trade.pnl_r < 0.0 {
                current_drawdown += trade.pnl_r;
                worst_overlap_drawdown = worst_overlap_drawdown.min(current_drawdown);
            } else {
                current_drawdown = 0.0; // Reset on win
            }
        }
        
        (pnl_by_overlap, worst_overlap_drawdown.abs(), overlap_1_pnl, overlap_2_pnl, overlap_3_pnl)
    }
}

// =============================================================================
// T5.2 — Regime-Based Reporting
// =============================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct RegimeReport {
    pub by_atr_regime: HashMap<String, RegimeBucket>,
    pub by_slope_regime: HashMap<String, RegimeBucket>,
    pub by_session: HashMap<String, RegimeBucket>,
    pub by_confidence_tier: HashMap<String, RegimeBucket>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct RegimeBucket {
    pub trades: u32,
    pub wins: u32,
    pub win_rate: f64,
    pub total_r: f64,
    pub avg_r: f64,
    pub expectancy: f64,
}

impl RegimeReport {
    pub fn generate(trades: &[TradeRecord]) -> Self {
        let mut report = Self::default();
        
        // Group trades by regimes
        let mut atr_groups: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        let mut slope_groups: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        let mut session_groups: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        let mut conf_groups: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        
        for trade in trades {
            // Confidence tier (always available)
            conf_groups.entry(trade.confidence_tier.clone())
                .or_default()
                .push(trade);
            
            // Regime context (optional)
            if let Some(ref regime) = trade.regime {
                atr_groups.entry(regime.atr_regime.clone())
                    .or_default()
                    .push(trade);
                slope_groups.entry(regime.slope_regime.clone())
                    .or_default()
                    .push(trade);
                session_groups.entry(regime.session.clone())
                    .or_default()
                    .push(trade);
            }
        }
        
        // Calculate buckets
        for (key, group) in atr_groups {
            report.by_atr_regime.insert(key, Self::calc_bucket(&group));
        }
        for (key, group) in slope_groups {
            report.by_slope_regime.insert(key, Self::calc_bucket(&group));
        }
        for (key, group) in session_groups {
            report.by_session.insert(key, Self::calc_bucket(&group));
        }
        for (key, group) in conf_groups {
            report.by_confidence_tier.insert(key, Self::calc_bucket(&group));
        }
        
        report
    }
    
    fn calc_bucket(trades: &[&TradeRecord]) -> RegimeBucket {
        let count = trades.len() as u32;
        if count == 0 {
            return RegimeBucket::default();
        }
        
        let wins = trades.iter().filter(|t| t.is_win).count() as u32;
        let win_rate = (wins as f64) / (count as f64) * 100.0;
        let total_r: f64 = trades.iter().map(|t| t.pnl_r).sum();
        let avg_r = total_r / count as f64;
        
        // Expectancy
        let avg_win = if wins > 0 {
            trades.iter().filter(|t| t.is_win).map(|t| t.pnl_r).sum::<f64>() / wins as f64
        } else { 0.0 };
        let losses = count - wins;
        let avg_loss = if losses > 0 {
            trades.iter().filter(|t| !t.is_win).map(|t| t.pnl_r.abs()).sum::<f64>() / losses as f64
        } else { 0.0 };
        let expectancy = ((wins as f64 / count as f64) * avg_win) - 
                         ((losses as f64 / count as f64) * avg_loss);
        
        RegimeBucket {
            trades: count,
            wins,
            win_rate,
            total_r,
            avg_r,
            expectancy,
        }
    }
}

// =============================================================================
// T4.1 — Centralized Penalty Engine
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PenaltyReason {
    // Hard blocks (score → 0)
    LowATR,
    FlatSlope,
    RecentSignal,
    WickTrap,
    BootstrapIncomplete,
    
    // Soft penalties
    NoDisplacement,
    NoLiquiditySweep,
    AgainstMajorTrend,
    WeakStructure,
    LowVolume,
    SessionOverlap,
    HighSpread,
    RecentVolatilitySpike,
}

impl PenaltyReason {
    pub fn penalty_value(&self) -> i32 {
        match self {
            // Hard blocks
            PenaltyReason::LowATR => -1000,
            PenaltyReason::RecentSignal => -1000,
            PenaltyReason::WickTrap => -1000,
            PenaltyReason::BootstrapIncomplete => -1000,
            
            // Soft penalties (optimized for more signals)
            PenaltyReason::FlatSlope => -25,           // Was hard block, now soft
            PenaltyReason::NoDisplacement => -8,       // Was -15
            PenaltyReason::NoLiquiditySweep => -5,     // Was -10
            PenaltyReason::AgainstMajorTrend => -20,   // Unchanged (important)
            PenaltyReason::WeakStructure => -6,        // Was -12
            PenaltyReason::LowVolume => -4,            // Was -8
            PenaltyReason::SessionOverlap => 0,        // Disabled
            PenaltyReason::HighSpread => -10,          // Unchanged (important)
            PenaltyReason::RecentVolatilitySpike => -8, // Was -15
        }
    }
    
    pub fn is_hard_block(&self) -> bool {
        self.penalty_value() <= -1000
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            PenaltyReason::LowATR => "ATR too low (< 0.8× avg)",
            PenaltyReason::FlatSlope => "EMA slope too flat",
            PenaltyReason::RecentSignal => "Cooldown active",
            PenaltyReason::WickTrap => "Wick trap detected",
            PenaltyReason::BootstrapIncomplete => "Bootstrap period incomplete",
            PenaltyReason::NoDisplacement => "No displacement candle",
            PenaltyReason::NoLiquiditySweep => "No liquidity sweep",
            PenaltyReason::AgainstMajorTrend => "Against major trend",
            PenaltyReason::WeakStructure => "Weak market structure",
            PenaltyReason::LowVolume => "Below average volume",
            PenaltyReason::SessionOverlap => "Low-activity session",
            PenaltyReason::HighSpread => "High spread detected",
            PenaltyReason::RecentVolatilitySpike => "Recent volatility spike",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PenaltyEngine {
    pub penalties: Vec<PenaltyReason>,
    pub base_score: i32,
}

impl PenaltyEngine {
    pub fn new(base_score: i32) -> Self {
        Self {
            penalties: Vec::new(),
            base_score,
        }
    }
    
    pub fn add_penalty(&mut self, reason: PenaltyReason) {
        self.penalties.push(reason);
    }
    
    pub fn has_hard_block(&self) -> bool {
        self.penalties.iter().any(|p| p.is_hard_block())
    }
    
    pub fn final_score(&self) -> i32 {
        if self.has_hard_block() {
            return 0;
        }
        
        let total_penalty: i32 = self.penalties.iter().map(|p| p.penalty_value()).sum();
        (self.base_score + total_penalty).max(0)
    }
    
    pub fn penalty_reasons(&self) -> Vec<String> {
        self.penalties.iter().map(|p| p.description().to_string()).collect()
    }
}

// =============================================================================
// T4.2 — Timeframe-Based Score Threshold
// =============================================================================

pub struct ScoreThreshold;

impl ScoreThreshold {
    /// Returns minimum score required for a signal on given timeframe
    /// OPTIMIZED: Lowered thresholds to allow more signals while maintaining quality
    pub fn min_score_for_tf(timeframe: &str) -> i32 {
        match timeframe {
            "5m" => 70,     // Was 80 - still high for noise
            "15m" => 65,    // Was 75
            "30m" => 60,    // Was 70
            "1h" => 55,     // Was 65
            "4h" => 55,     // Was 65  
            "1d" => 50,     // Was 60
            _ => 60,        // Was 70
        }
    }
    
    /// Check if score meets threshold
    pub fn passes_threshold(timeframe: &str, score: i32) -> bool {
        score >= Self::min_score_for_tf(timeframe)
    }
}

// =============================================================================
// T6.1 — Confidence-Based Routing
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceTier {
    High,   // 80-100: Full size, immediate execution
    Medium, // 65-79:  Reduced size, standard execution
    Low,    // < 65:   Skip or paper-trade only
}

impl ConfidenceTier {
    pub fn from_score(score: i32) -> Self {
        match score {
            80..=100 => ConfidenceTier::High,
            65..=79 => ConfidenceTier::Medium,
            _ => ConfidenceTier::Low,
        }
    }
    
    pub fn position_size_multiplier(&self) -> f64 {
        match self {
            ConfidenceTier::High => 1.0,
            ConfidenceTier::Medium => 0.5,
            ConfidenceTier::Low => 0.0, // Don't trade
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            ConfidenceTier::High => "high",
            ConfidenceTier::Medium => "medium",
            ConfidenceTier::Low => "low",
        }
    }
}

// =============================================================================
// Regime Determination Helpers
// =============================================================================

impl RegimeContext {
    pub fn determine(
        current_atr: Decimal,
        avg_atr: Decimal,
        ema_slope: Decimal,
        hour_utc: u32,
    ) -> Self {
        // ATR Regime
        let atr_ratio = if avg_atr > Decimal::ZERO {
            current_atr / avg_atr
        } else {
            Decimal::ONE
        };
        
        let atr_regime = if atr_ratio < Decimal::new(8, 1) {
            "low"
        } else if atr_ratio > Decimal::new(15, 1) {
            "high"
        } else {
            "normal"
        }.to_string();
        
        // Slope Regime
        let slope_abs = ema_slope.abs();
        let slope_regime = if slope_abs < Decimal::new(1, 3) { // < 0.001
            "flat"
        } else if ema_slope > Decimal::ZERO {
            "trending_up"
        } else {
            "trending_down"
        }.to_string();
        
        // Session by UTC hour
        let session = match hour_utc {
            0..=7 => "asia",      // 00:00 - 07:59 UTC
            8..=15 => "london",   // 08:00 - 15:59 UTC
            16..=23 => "ny",      // 16:00 - 23:59 UTC
            _ => "unknown",
        }.to_string();
        
        Self {
            atr_regime,
            slope_regime,
            session,
        }
    }
}

// =============================================================================
// Extended Backtest Result
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ExtendedBacktestResult {
    pub symbol: String,
    pub timeframe: String,
    pub total_trades: u32,
    pub wins: u32,
    pub losses: u32,
    pub win_rate: f64,
    pub total_pnl_r: f64,
    pub advanced_metrics: AdvancedMetrics,
    pub regime_report: RegimeReport,
    pub signals: Vec<TradeSignal>,
}
