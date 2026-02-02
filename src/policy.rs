// ============================================================
// PHASE 0 & 1 & 3 — Policy, Bootstrap, Regime & Asset Filters
// ============================================================

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet};
use crate::types::Candle;
use crate::analytics::{PenaltyEngine, PenaltyReason, ScoreThreshold};

// ============================================================
// T0.1 — Timeframe Policy Enforcement
// ============================================================

/// Asset/Timeframe policy table
/// Returns false for disallowed combinations
pub struct TimeframePolicy {
    blocked: HashSet<(String, String)>, // (symbol, timeframe)
}

impl TimeframePolicy {
    pub fn new() -> Self {
        let mut blocked = HashSet::new();
        
        // BTC: ❌ 5m
        blocked.insert(("BTCUSDT".to_string(), "5m".to_string()));
        
        // ETH: ❌ 1d
        blocked.insert(("ETHUSDT".to_string(), "1d".to_string()));
        
        // SOL: ❌ 5m, ❌ 30m
        blocked.insert(("SOLUSDT".to_string(), "5m".to_string()));
        blocked.insert(("SOLUSDT".to_string(), "30m".to_string()));
        
        Self { blocked }
    }
    
    /// Check if symbol/timeframe combination is allowed for trading
    pub fn is_allowed(&self, symbol: &str, timeframe: &str) -> bool {
        !self.blocked.contains(&(symbol.to_string(), timeframe.to_string()))
    }
    
    /// Get rejection reason if blocked
    pub fn get_block_reason(&self, symbol: &str, timeframe: &str) -> Option<String> {
        if self.blocked.contains(&(symbol.to_string(), timeframe.to_string())) {
            Some(format!("Policy violation: {} {} is blocked for trading", symbol, timeframe))
        } else {
            None
        }
    }
}

impl Default for TimeframePolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T0.2 — Bootstrap Integrity Gate
// ============================================================

/// Bootstrap state tracker
#[derive(Debug, Clone)]
pub struct BootstrapState {
    pub ema200_ready: bool,      // EMA200 needs 200+ candles to seed properly
    pub pivot_history_ready: bool, // Need minimum pivot history
    pub atr_ready: bool,          // ATR needs warmup
    pub candle_count: usize,
    pub suppression_reason: Option<String>,
}

impl BootstrapState {
    pub fn new() -> Self {
        Self {
            ema200_ready: false,
            pivot_history_ready: false,
            atr_ready: false,
            candle_count: 0,
            suppression_reason: None,
        }
    }
    
    /// Update bootstrap state based on candle count and indicator states
    pub fn update(&mut self, candle_count: usize, has_ema200: bool, pivot_count: usize, has_atr: bool) {
        self.candle_count = candle_count;
        
        // EMA200 needs at least 200 candles for proper seeding
        self.ema200_ready = candle_count >= 200 && has_ema200;
        
        // Pivot history needs at least 2 pivots on each side for structure
        self.pivot_history_ready = pivot_count >= 2;
        
        // ATR needs at least 14 candles
        self.atr_ready = candle_count >= 14 && has_atr;
        
        // Set suppression reason if not ready
        if !self.is_complete() {
            let mut reasons = Vec::new();
            if !self.ema200_ready { reasons.push("EMA200 not seeded"); }
            if !self.pivot_history_ready { reasons.push("Insufficient pivot history"); }
            if !self.atr_ready { reasons.push("ATR not ready"); }
            self.suppression_reason = Some(format!("bootstrap_incomplete: {}", reasons.join(", ")));
        } else {
            self.suppression_reason = None;
        }
    }
    
    /// Check if bootstrap is complete and signals can be generated
    pub fn is_complete(&self) -> bool {
        self.ema200_ready && self.pivot_history_ready && self.atr_ready
    }
}

impl Default for BootstrapState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T1.1 — Volatility Regime Filter (ATR-based)
// ============================================================

#[derive(Debug, Clone)]
pub struct VolatilityFilter {
    /// Minimum ATR ratio threshold by timeframe
    min_thresholds: HashMap<String, Decimal>,
    /// Extreme low ATR multiplier (below this = hard block)
    extreme_low_multiplier: Decimal,
}

impl VolatilityFilter {
    pub fn new() -> Self {
        let mut min_thresholds = HashMap::new();
        
        // Timeframe-specific minimum ATR ratios
        min_thresholds.insert("5m".to_string(), Decimal::from_f64(0.0012).unwrap());
        min_thresholds.insert("15m".to_string(), Decimal::from_f64(0.0015).unwrap());
        min_thresholds.insert("30m".to_string(), Decimal::from_f64(0.0012).unwrap());
        min_thresholds.insert("1h".to_string(), Decimal::from_f64(0.0010).unwrap());
        min_thresholds.insert("4h".to_string(), Decimal::from_f64(0.0008).unwrap());
        min_thresholds.insert("1d".to_string(), Decimal::from_f64(0.0006).unwrap());
        
        Self {
            min_thresholds,
            extreme_low_multiplier: Decimal::from_f64(0.5).unwrap(), // 50% of threshold = extreme
        }
    }
    
    /// Evaluate volatility regime
    /// Returns: (score_penalty, hard_block, reason)
    pub fn evaluate(&self, timeframe: &str, atr_ratio: Decimal, median_ratio: Decimal) -> (i32, bool, Option<String>) {
        let threshold = self.min_thresholds
            .get(timeframe)
            .copied()
            .unwrap_or(Decimal::from_f64(0.0010).unwrap());
        
        let extreme_threshold = threshold * self.extreme_low_multiplier;
        
        // Hard block: Extreme low volatility
        if atr_ratio < extreme_threshold {
            return (-15, true, Some(format!(
                "🚫 HARD BLOCK: Extreme low volatility (ATR ratio {:.4} < {:.4})",
                atr_ratio, extreme_threshold
            )));
        }
        
        // Soft penalty: Below threshold or below median
        if atr_ratio < threshold || atr_ratio < median_ratio {
            return (-15, false, Some(format!(
                "⚠️ Low volatility regime: ATR ratio {:.4} (threshold: {:.4}, median: {:.4}) (-15)",
                atr_ratio, threshold, median_ratio
            )));
        }
        
        (0, false, None)
    }
}

impl Default for VolatilityFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T1.2 — EMA Slope Filter
// ============================================================

#[derive(Debug, Clone)]
pub struct SlopeFilter {
    /// Minimum slope threshold (percentage)
    min_slope: Decimal,
}

impl SlopeFilter {
    pub fn new() -> Self {
        Self {
            min_slope: Decimal::from_f64(0.0006).unwrap(), // 0.06%
        }
    }
    
    /// Evaluate EMA slope
    /// Returns: (score_penalty, hard_block_with_low_atr, reason)
    pub fn evaluate(&self, slope: Decimal, is_low_atr: bool) -> (i32, bool, Option<String>) {
        let abs_slope = slope.abs();
        
        if abs_slope < self.min_slope {
            // Hard block if combined with low ATR
            if is_low_atr {
                return (-10, true, Some(format!(
                    "🚫 HARD BLOCK: Flat EMA + Low ATR (slope: {:.6})",
                    slope
                )));
            }
            
            return (-10, false, Some(format!(
                "⚠️ Flat EMA slope = chop market (slope: {:.6}) (-10)",
                slope
            )));
        }
        
        (0, false, None)
    }
}

impl Default for SlopeFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T1.3 — Cooldown Enforcement (Context-Based for Multi-Position)
// ============================================================

use crate::types::ContextId;

#[derive(Debug, Clone)]
pub struct CooldownManager {
    /// Last signal candle index per symbol/timeframe
    last_signals: HashMap<String, usize>,
    /// Last trade close candle index per symbol/timeframe (LEGACY - kept for single trade mode)
    last_trade_closes: HashMap<String, usize>,
    /// Context-based cooldowns: context_id -> close_candle_idx
    context_cooldowns: HashMap<String, usize>,
    /// Minimum candles between trade closes
    min_cooldown: usize,
    /// Backtest mode flag - shorter cooldowns
    is_backtest_mode: bool,
}

impl CooldownManager {
    pub fn new() -> Self {
        Self {
            last_signals: HashMap::new(),
            last_trade_closes: HashMap::new(),
            context_cooldowns: HashMap::new(),
            min_cooldown: 2, // Reduced from 3
            is_backtest_mode: false,
        }
    }
    
    /// Create with backtest mode
    pub fn with_backtest_mode(backtest: bool) -> Self {
        Self {
            last_signals: HashMap::new(),
            last_trade_closes: HashMap::new(),
            context_cooldowns: HashMap::new(),
            min_cooldown: if backtest { 1 } else { 2 }, // 1 candle in backtest, 2 in live
            is_backtest_mode: backtest,
        }
    }
    
    /// Get cooldown duration based on timeframe
    pub fn get_cooldown_for_tf(&self, timeframe: &str) -> usize {
        let base = if self.is_backtest_mode { 1 } else { self.min_cooldown };
        
        match timeframe {
            "5m" | "15m" => base + 1,  // 2-3 candles for lower TF
            "30m" | "1h" => base,      // 1-2 candles for mid TF
            "4h" | "1d" => 1,          // 1 candle for higher TF (always)
            _ => base
        }
    }
    
    /// Check if a specific context is on cooldown (MULTI-POSITION MODE)
    /// Only blocks the SAME context, not all trades
    pub fn is_context_on_cooldown(&self, context_id: &ContextId, timeframe: &str, current_candle_idx: usize) -> bool {
        let key = context_id.to_string();
        
        if let Some(&close_idx) = self.context_cooldowns.get(&key) {
            let cooldown = self.get_cooldown_for_tf(timeframe);
            let elapsed = current_candle_idx.saturating_sub(close_idx);
            elapsed < cooldown
        } else {
            false
        }
    }
    
    /// Record context-based trade close (MULTI-POSITION MODE)
    /// Cooldown now applies only to the same context, not globally
    pub fn record_context_close(&mut self, context_id: &ContextId, candle_idx: usize) {
        self.context_cooldowns.insert(context_id.to_string(), candle_idx);
    }
    
    /// Check if cooldown is active - NOW BASED ON TRADE CLOSE, NOT SIGNAL
    /// Returns true ONLY during the cooldown period AFTER a trade closes
    /// Does NOT return true when a trade is open - that's handled by has_open_trade()
    pub fn is_on_cooldown(&self, key: &str, current_candle_idx: usize) -> bool {
        // Cooldown is ONLY active between trade close and cooldown expiry
        // If no trade has ever closed for this key, no cooldown
        if let Some(&close_idx) = self.last_trade_closes.get(key) {
            // Check if there's a NEW signal after the close (means new trade opened)
            // If so, we're not in cooldown - we're in a new trade
            if let Some(&signal_idx) = self.last_signals.get(key) {
                if signal_idx > close_idx {
                    // New signal came after close = new trade is open, no cooldown
                    return false;
                }
            }
            
            // No new signal after close = check actual cooldown period
            let tf = key.split('_').nth(1).unwrap_or("1h");
            let cooldown = self.get_cooldown_for_tf(tf);
            
            let elapsed = current_candle_idx.saturating_sub(close_idx);
            elapsed < cooldown
        } else {
            // No trade ever closed = no cooldown
            false
        }
    }
    
    /// Check if there's an active trade (signal sent, not yet closed) - LEGACY for single-position
    pub fn has_open_trade(&self, key: &str) -> bool {
        if let Some(&signal_idx) = self.last_signals.get(key) {
            if let Some(&close_idx) = self.last_trade_closes.get(key) {
                signal_idx > close_idx
            } else {
                true
            }
        } else {
            false
        }
    }
    
    /// Record a signal (trade open)
    pub fn record_signal(&mut self, key: &str, candle_idx: usize) {
        self.last_signals.insert(key.to_string(), candle_idx);
    }
    
    /// Record trade close (TP/SL/BE hit) - THIS STARTS COOLDOWN
    pub fn record_trade_close(&mut self, key: &str, candle_idx: usize) {
        self.last_trade_closes.insert(key.to_string(), candle_idx);
    }
    
    /// Get remaining cooldown candles
    pub fn remaining_cooldown(&self, key: &str, current_candle_idx: usize) -> usize {
        if let Some(&close_idx) = self.last_trade_closes.get(key) {
            let tf = key.split('_').nth(1).unwrap_or("1h");
            let cooldown = self.get_cooldown_for_tf(tf);
            let elapsed = current_candle_idx.saturating_sub(close_idx);
            if elapsed < cooldown {
                return cooldown - elapsed;
            }
        }
        0
    }
    
    /// Get remaining cooldown for a specific context
    pub fn remaining_context_cooldown(&self, context_id: &ContextId, timeframe: &str, current_candle_idx: usize) -> usize {
        let key = context_id.to_string();
        if let Some(&close_idx) = self.context_cooldowns.get(&key) {
            let cooldown = self.get_cooldown_for_tf(timeframe);
            let elapsed = current_candle_idx.saturating_sub(close_idx);
            if elapsed < cooldown {
                return cooldown - elapsed;
            }
        }
        0
    }
}

impl Default for CooldownManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T2.1 — Liquidity-aware BOS
// ============================================================

#[derive(Debug, Clone)]
pub struct LiquidityBosAnalyzer {
    /// Tolerance for equal high/low detection (percentage)
    equal_tolerance: Decimal,
    /// ATR multiplier for displacement detection
    displacement_atr_mult: Decimal,
}

impl LiquidityBosAnalyzer {
    pub fn new() -> Self {
        Self {
            equal_tolerance: Decimal::from_f64(0.0015).unwrap(), // 0.15%
            displacement_atr_mult: Decimal::from_f64(1.2).unwrap(),
        }
    }
    
    /// Analyze BOS quality
    /// Returns: (score_adjustment, is_strong_bos, reason)
    pub fn evaluate(
        &self,
        bos_confirmed: bool,
        has_equal_highs: bool,
        has_equal_lows: bool,
        has_displacement: bool,
        is_bullish: bool,
    ) -> (i32, bool, Option<String>) {
        if !bos_confirmed {
            return (0, false, None);
        }
        
        let has_liquidity = if is_bullish { has_equal_lows } else { has_equal_highs };
        
        // Strong BOS: Liquidity sweep OR displacement
        if has_liquidity && has_displacement {
            return (15, true, Some("✅ Strong BOS: Liquidity sweep + Displacement (+15)".to_string()));
        }
        
        if has_liquidity {
            return (10, true, Some("✅ BOS with liquidity sweep (+10)".to_string()));
        }
        
        if has_displacement {
            return (10, true, Some("✅ BOS with strong displacement (+10)".to_string()));
        }
        
        // Weak BOS: No liquidity, no displacement
        (-10, false, Some("⚠️ Weak BOS: No displacement, no liquidity sweep (-10)".to_string()))
    }
}

impl Default for LiquidityBosAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T2.2 — Pivot + Displacement Validation
// ============================================================

#[derive(Debug, Clone)]
pub struct DisplacementValidator {
    /// Minimum body/range ratio for strong displacement
    min_body_ratio: Decimal,
}

impl DisplacementValidator {
    pub fn new() -> Self {
        Self {
            min_body_ratio: Decimal::from_f64(0.65).unwrap(),
        }
    }
    
    /// Validate candle displacement after pivot
    /// Returns: (score_adjustment, reason)
    pub fn evaluate(&self, candle: &Candle, is_bullish_signal: bool) -> (i32, Option<String>) {
        let range = candle.high - candle.low;
        if range.is_zero() {
            return (-10, Some("⚠️ Zero range candle after pivot (-10)".to_string()));
        }
        
        let body = (candle.close - candle.open).abs();
        let body_ratio = body / range;
        
        // Check directional close
        let directional = if is_bullish_signal {
            candle.close > candle.open // Bullish candle
        } else {
            candle.close < candle.open // Bearish candle
        };
        
        if body_ratio >= self.min_body_ratio && directional {
            return (10, Some(format!(
                "✅ Strong displacement: body/range {:.1}% (+10)",
                body_ratio * Decimal::from(100)
            )));
        }
        
        if body_ratio < Decimal::from_f64(0.4).unwrap() {
            return (-10, Some(format!(
                "⚠️ Weak displacement/indecision: body/range {:.1}% (-10)",
                body_ratio * Decimal::from(100)
            )));
        }
        
        (0, None)
    }
}

impl Default for DisplacementValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T3.1 — Wick Trap Filter (SOL-focused)
// ============================================================

#[derive(Debug, Clone)]
pub struct WickTrapFilter {
    /// Maximum wick ratio before blocking
    max_wick_ratio: Decimal,
    /// Symbols where this filter is active
    active_symbols: HashSet<String>,
}

impl WickTrapFilter {
    pub fn new() -> Self {
        let mut active_symbols = HashSet::new();
        active_symbols.insert("SOLUSDT".to_string());
        
        Self {
            max_wick_ratio: Decimal::from_f64(0.5).unwrap(),
            active_symbols,
        }
    }
    
    /// Check for wick trap
    /// Returns: (hard_block, reason)
    pub fn evaluate(&self, symbol: &str, candle: &Candle) -> (bool, Option<String>) {
        if !self.active_symbols.contains(symbol) {
            return (false, None);
        }
        
        let range = candle.high - candle.low;
        if range.is_zero() {
            return (false, None);
        }
        
        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);
        
        let upper_wick = candle.high - body_top;
        let lower_wick = body_bottom - candle.low;
        let max_wick = upper_wick.max(lower_wick);
        
        let wick_ratio = max_wick / range;
        
        if wick_ratio > self.max_wick_ratio {
            return (true, Some(format!(
                "🚫 HARD BLOCK: Wick trap detected on {} (wick ratio: {:.1}% > 50%)",
                symbol, wick_ratio * Decimal::from(100)
            )));
        }
        
        (false, None)
    }
}

impl Default for WickTrapFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// T3.2 — ETH Micro-Boost
// ============================================================

#[derive(Debug, Clone)]
pub struct EthMicroBoost;

impl EthMicroBoost {
    pub fn new() -> Self {
        Self
    }
    
    /// Apply ETH-specific boost
    /// Returns: (score_bonus, reason)
    pub fn evaluate(
        &self,
        symbol: &str,
        has_liquidity_sweep: bool,
        price: Decimal,
        ema13: Option<Decimal>,
        ema50: Option<Decimal>,
        is_bullish: bool,
    ) -> (i32, Option<String>) {
        if symbol != "ETHUSDT" {
            return (0, None);
        }
        
        // Check EMA confluence
        let ema_confluence = match (ema13, ema50) {
            (Some(e13), Some(e50)) => {
                if is_bullish {
                    price > e13 && price > e50 && e13 > e50
                } else {
                    price < e13 && price < e50 && e13 < e50
                }
            }
            _ => false,
        };
        
        if has_liquidity_sweep && ema_confluence {
            return (5, Some("✅ ETH Micro-Boost: Liquidity sweep + EMA confluence (+5)".to_string()));
        }
        
        (0, None)
    }
}

impl Default for EthMicroBoost {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Combined Filter Result
// ============================================================

#[derive(Debug, Clone)]
pub struct FilterResult {
    pub allowed: bool,
    pub hard_blocked: bool,
    pub score_adjustment: i32,
    pub reasons: Vec<String>,
    pub suppression_reason: Option<String>,
}

impl FilterResult {
    pub fn new() -> Self {
        Self {
            allowed: true,
            hard_blocked: false,
            score_adjustment: 0,
            reasons: Vec::new(),
            suppression_reason: None,
        }
    }
    
    pub fn block(&mut self, reason: String) {
        self.allowed = false;
        self.hard_blocked = true;
        self.suppression_reason = Some(reason.clone());
        self.reasons.push(reason);
    }
    
    pub fn add_penalty(&mut self, penalty: i32, reason: String) {
        self.score_adjustment += penalty;
        self.reasons.push(reason);
    }
    
    pub fn add_bonus(&mut self, bonus: i32, reason: String) {
        self.score_adjustment += bonus;
        self.reasons.push(reason);
    }
}

impl Default for FilterResult {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Master Policy Engine
// ============================================================

pub struct PolicyEngine {
    pub timeframe_policy: TimeframePolicy,
    pub volatility_filter: VolatilityFilter,
    pub slope_filter: SlopeFilter,
    pub cooldown_manager: CooldownManager,
    pub liquidity_bos: LiquidityBosAnalyzer,
    pub displacement_validator: DisplacementValidator,
    pub wick_trap_filter: WickTrapFilter,
    pub eth_micro_boost: EthMicroBoost,
}

impl PolicyEngine {
    pub fn new() -> Self {
        Self {
            timeframe_policy: TimeframePolicy::new(),
            volatility_filter: VolatilityFilter::new(),
            slope_filter: SlopeFilter::new(),
            cooldown_manager: CooldownManager::new(),
            liquidity_bos: LiquidityBosAnalyzer::new(),
            displacement_validator: DisplacementValidator::new(),
            wick_trap_filter: WickTrapFilter::new(),
            eth_micro_boost: EthMicroBoost::new(),
        }
    }
    
    /// T1 Cooldown Fix: Create engine with backtest mode (shorter cooldowns)
    pub fn new_backtest_mode() -> Self {
        Self {
            timeframe_policy: TimeframePolicy::new(),
            volatility_filter: VolatilityFilter::new(),
            slope_filter: SlopeFilter::new(),
            cooldown_manager: CooldownManager::with_backtest_mode(true),
            liquidity_bos: LiquidityBosAnalyzer::new(),
            displacement_validator: DisplacementValidator::new(),
            wick_trap_filter: WickTrapFilter::new(),
            eth_micro_boost: EthMicroBoost::new(),
        }
    }
    
    /// T4.1: Run all filters through centralized PenaltyEngine
    /// Returns: (final_score, should_block, reasons)
    pub fn evaluate_with_penalties(
        &mut self,
        base_score: i32,
        symbol: &str,
        timeframe: &str,
        candle: &Candle,
        atr_ratio: Decimal,
        median_atr_ratio: Decimal,
        ema_slope: Decimal,
        candle_idx: usize,
        bootstrap_complete: bool,
        has_bos: bool,
        has_equal_highs: bool,
        has_equal_lows: bool,
        has_displacement: bool,
        is_bullish: bool,
        ema13: Option<Decimal>,
        ema50: Option<Decimal>,
    ) -> (i32, bool, Vec<String>) {
        let mut penalty_engine = PenaltyEngine::new(base_score);
        let mut reasons = Vec::new();
        
        // T0.1: Timeframe Policy
        if let Some(reason) = self.timeframe_policy.get_block_reason(symbol, timeframe) {
            return (0, true, vec![reason]);
        }
        
        // T0.2: Bootstrap check
        if !bootstrap_complete {
            penalty_engine.add_penalty(PenaltyReason::BootstrapIncomplete);
        }
        
        // T1.1: Volatility filter
        let (vol_penalty, vol_block, vol_reason) = self.volatility_filter.evaluate(
            timeframe, atr_ratio, median_atr_ratio
        );
        if vol_block {
            penalty_engine.add_penalty(PenaltyReason::LowATR);
        }
        if let Some(r) = vol_reason {
            reasons.push(r);
        }
        
        // T1.2: Slope filter
        let is_low_atr = atr_ratio < Decimal::from_f64(0.001).unwrap();
        let (slope_penalty, slope_block, slope_reason) = self.slope_filter.evaluate(ema_slope, is_low_atr);
        if slope_block {
            penalty_engine.add_penalty(PenaltyReason::FlatSlope);
        }
        if let Some(r) = slope_reason {
            reasons.push(r);
        }
        
        // T1.3: Cooldown
        let key = format!("{}_{}", symbol, timeframe);
        if self.cooldown_manager.is_on_cooldown(&key, candle_idx) {
            penalty_engine.add_penalty(PenaltyReason::RecentSignal);
            reasons.push(format!(
                "🚫 Cooldown active: {} candles remaining",
                self.cooldown_manager.remaining_cooldown(&key, candle_idx)
            ));
        }
        
        // T3.1: Wick trap (SOL)
        let (wick_block, wick_reason) = self.wick_trap_filter.evaluate(symbol, candle);
        if wick_block {
            penalty_engine.add_penalty(PenaltyReason::WickTrap);
        }
        if let Some(r) = wick_reason {
            reasons.push(r);
        }
        
        // Check for hard blocks early
        if penalty_engine.has_hard_block() {
            return (0, true, penalty_engine.penalty_reasons());
        }
        
        // T2.1: Liquidity-aware BOS (score adjustments)
        let (bos_adj, _strong_bos, bos_reason) = self.liquidity_bos.evaluate(
            has_bos, has_equal_highs, has_equal_lows, has_displacement, is_bullish
        );
        if bos_adj < 0 {
            penalty_engine.add_penalty(PenaltyReason::NoLiquiditySweep);
        }
        if let Some(r) = bos_reason {
            reasons.push(r);
        }
        
        // T2.2: Displacement validation
        let (disp_adj, disp_reason) = self.displacement_validator.evaluate(candle, is_bullish);
        if disp_adj < 0 {
            penalty_engine.add_penalty(PenaltyReason::NoDisplacement);
        }
        if let Some(r) = disp_reason {
            reasons.push(r);
        }
        
        // T3.2: ETH micro-boost
        let has_liquidity = if is_bullish { has_equal_lows } else { has_equal_highs };
        let (eth_bonus, eth_reason) = self.eth_micro_boost.evaluate(
            symbol, has_liquidity, candle.close, ema13, ema50, is_bullish
        );
        if let Some(r) = eth_reason {
            reasons.push(r);
        }
        
        // Calculate final score with bonuses
        let mut final_score = penalty_engine.final_score();
        final_score += bos_adj.max(0);  // Only add positive BOS adjustments
        final_score += disp_adj.max(0); // Only add positive displacement adjustments
        final_score += eth_bonus;
        
        // T4.2: Check timeframe threshold
        let passes_threshold = ScoreThreshold::passes_threshold(timeframe, final_score);
        if !passes_threshold {
            reasons.push(format!(
                "⚠️ Score {} below {} threshold ({})",
                final_score, timeframe, ScoreThreshold::min_score_for_tf(timeframe)
            ));
        }
        
        (final_score, !passes_threshold, reasons)
    }
    
    /// Record signal for cooldown tracking
    pub fn record_signal(&mut self, symbol: &str, timeframe: &str, candle_idx: usize) {
        let key = format!("{}_{}", symbol, timeframe);
        self.cooldown_manager.record_signal(&key, candle_idx);
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timeframe_policy() {
        let policy = TimeframePolicy::new();
        
        // Blocked combinations
        assert!(!policy.is_allowed("BTCUSDT", "5m"));
        assert!(!policy.is_allowed("ETHUSDT", "1d"));
        assert!(!policy.is_allowed("SOLUSDT", "5m"));
        assert!(!policy.is_allowed("SOLUSDT", "30m"));
        
        // Allowed combinations
        assert!(policy.is_allowed("BTCUSDT", "15m"));
        assert!(policy.is_allowed("BTCUSDT", "1h"));
        assert!(policy.is_allowed("ETHUSDT", "15m"));
        assert!(policy.is_allowed("SOLUSDT", "15m"));
        assert!(policy.is_allowed("SOLUSDT", "1h"));
    }
    
    #[test]
    fn test_bootstrap_state() {
        let mut state = BootstrapState::new();
        
        // Initial state - not complete
        assert!(!state.is_complete());
        
        // Partial update
        state.update(100, true, 1, true);
        assert!(!state.is_complete()); // EMA200 not ready (need 200 candles)
        
        // Full update
        state.update(250, true, 3, true);
        assert!(state.is_complete());
    }
}
