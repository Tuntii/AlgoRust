use chrono::{DateTime, Utc};
use rust_decimal::prelude::FromStr;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// MULTI-POSITION TRADING TYPES
// =============================================================================

/// Unique identifier for a trading context (BOS, liquidity sweep, etc.)
/// This prevents entering the same setup multiple times
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextId {
    /// Type of context: "bos", "liquidity_sweep", "pivot"
    pub context_type: String,
    /// Unique identifier (e.g., pivot timestamp, BOS index)
    pub identifier: String,
    /// The candle index when this context was created
    pub created_at_candle: usize,
}

impl ContextId {
    pub fn new(context_type: &str, identifier: &str, candle_idx: usize) -> Self {
        Self {
            context_type: context_type.to_string(),
            identifier: identifier.to_string(),
            created_at_candle: candle_idx,
        }
    }

    /// Create context ID from BOS event
    pub fn from_bos(candle_idx: usize, is_bullish: bool) -> Self {
        Self::new(
            "bos",
            &format!(
                "{}_{}",
                if is_bullish { "bull" } else { "bear" },
                candle_idx
            ),
            candle_idx,
        )
    }

    /// Create context ID from liquidity sweep
    pub fn from_liquidity_sweep(candle_idx: usize, level: Decimal) -> Self {
        Self::new(
            "liquidity_sweep",
            &format!("{}_{}", level, candle_idx),
            candle_idx,
        )
    }

    /// Create context ID from pivot confirmation
    pub fn from_pivot(candle_idx: usize, is_high: bool) -> Self {
        Self::new(
            "pivot",
            &format!("{}_{}", if is_high { "high" } else { "low" }, candle_idx),
            candle_idx,
        )
    }
}

impl fmt::Display for ContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.context_type, self.identifier)
    }
}

/// Active trade in the position pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTrade {
    /// Unique trade ID
    pub trade_id: String,
    /// The signal that triggered this trade
    pub signal: TradeSignal,
    /// Entry price
    pub entry_price: Decimal,
    /// Stop loss price (can be modified for BE)
    pub sl_price: Decimal,
    /// Original stop loss price (for BE tracking)
    pub original_sl_price: Decimal,
    /// Take profit price
    pub tp_price: Decimal,
    /// Direction of the trade
    pub direction: SignalType,
    /// Context ID that generated this trade (for uniqueness check)
    pub context_id: ContextId,
    /// Candle index when trade was opened
    pub opened_at_candle: usize,
    /// Exit price (if closed)
    pub exit_price: Option<Decimal>,
    /// PnL in R (if closed)
    pub pnl_r: Option<Decimal>,
    /// Outcome: "WIN", "LOSS", "BE", "MAX_DURATION", "PARTIAL" (if closed)
    pub outcome: Option<String>,
    /// Candle index when trade was closed
    pub exit_candle_idx: Option<usize>,
    /// Duration in candles
    pub duration_candles: Option<u32>,
    /// Confidence adjustment based on concurrent trades
    pub adjusted_confidence: u8,
    /// T8.2: Context strength score for ranking (higher = stronger)
    pub context_score: i32,
    /// T8.3: EMA50 slope at entry (for trend saturation check)
    pub ema50_slope_at_entry: Option<Decimal>,
    /// T9.2: Whether SL has been moved to BE
    pub is_be_applied: bool,
    /// T9.3: Partial TP taken (50% closed at 1R)
    pub partial_tp_taken: bool,
}

impl ActiveTrade {
    pub fn new(
        signal: TradeSignal,
        entry_price: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
        context_id: ContextId,
        opened_at_candle: usize,
    ) -> Self {
        use uuid::Uuid;

        let direction = signal.signal.clone();
        let confidence = signal.confidence;

        Self {
            trade_id: Uuid::new_v4().to_string(),
            signal,
            entry_price,
            sl_price,
            original_sl_price: sl_price, // Store original SL for BE reference
            tp_price,
            direction,
            context_id,
            opened_at_candle,
            exit_price: None,
            pnl_r: None,
            outcome: None,
            exit_candle_idx: None,
            duration_candles: None,
            adjusted_confidence: confidence,
            context_score: 0,           // Will be set externally
            ema50_slope_at_entry: None, // Will be set externally
            is_be_applied: false,
            partial_tp_taken: false,
        }
    }

    /// Create with context score (T8.2)
    pub fn with_context_score(mut self, score: i32) -> Self {
        self.context_score = score;
        self
    }

    /// Set EMA50 slope at entry (T8.3)
    pub fn with_ema50_slope(mut self, slope: Decimal) -> Self {
        self.ema50_slope_at_entry = Some(slope);
        self
    }

    /// Apply break-even to this trade (T9.2)
    pub fn apply_be(&mut self) {
        if !self.is_be_applied {
            self.sl_price = self.entry_price;
            self.is_be_applied = true;
        }
    }

    /// Check if trade is still open
    pub fn is_open(&self) -> bool {
        self.outcome.is_none()
    }

    /// Get current duration in candles
    pub fn current_duration(&self, current_candle_idx: usize) -> u32 {
        if self.opened_at_candle <= current_candle_idx {
            (current_candle_idx - self.opened_at_candle) as u32
        } else {
            0
        }
    }

    /// Calculate unrealized R at given price
    pub fn unrealized_r(&self, current_price: Decimal) -> Decimal {
        let risk = (self.entry_price - self.original_sl_price).abs();
        if risk.is_zero() {
            return Decimal::ZERO;
        }

        match self.direction {
            SignalType::LONG => (current_price - self.entry_price) / risk,
            SignalType::SHORT => (self.entry_price - current_price) / risk,
        }
    }

    /// Close the trade with given outcome
    pub fn close(
        &mut self,
        exit_price: Decimal,
        pnl_r: Decimal,
        outcome: &str,
        exit_candle_idx: usize,
    ) {
        self.exit_price = Some(exit_price);
        self.pnl_r = Some(pnl_r);
        self.outcome = Some(outcome.to_string());
        self.exit_candle_idx = Some(exit_candle_idx);
        self.duration_candles = Some((exit_candle_idx - self.opened_at_candle) as u32);
    }
}

/// Multi-position pool configuration
#[derive(Debug, Clone)]
pub struct PositionPoolConfig {
    /// Maximum active trades per symbol/timeframe (T8.1 HARD CAP)
    pub max_active_trades_per_symbol_tf: usize,
    /// Allow same direction trades (LONG + LONG)
    pub allow_same_direction: bool,
    /// Allow opposite direction trades (LONG + SHORT) - ALWAYS FALSE per strategy
    pub allow_hedge: bool,
    /// Confidence reduction per additional trade
    pub confidence_reduction_per_trade: f64,
    /// T8.2: Minimum score improvement to replace a weaker trade
    pub min_score_improvement: i32,
    /// T9.1: Max trade duration in candles before forced exit
    pub max_trade_duration_candles: u32,
    /// T9.2: Duration threshold (candles) before BE can be applied
    pub be_threshold_candles: u32,
    /// T9.2: Min unrealized R to NOT apply BE (above this, let it ride)
    pub be_min_profit_r: Decimal,
    /// T9.3: Enable partial TP at 1R
    pub partial_tp_enabled: bool,
    /// T9.3: Partial TP ratio (e.g., 0.5 = close 50%)
    pub partial_tp_ratio: f64,
    /// T10.2: Kill switch - max consecutive losses before pause
    pub kill_switch_consec_losses: u32,
    /// T11.1: Kill switch minimum duration (candles) before it can be reset
    pub kill_switch_min_duration: u32,
    /// T11.2: Consecutive wins needed to reset kill switch
    pub kill_switch_reset_wins: u32,
}

// =============================================================================
// T11 - KILL SWITCH STATE (PER SYMBOL+TF)
// =============================================================================

/// T11.3: Kill switch state for a specific symbol+timeframe
#[derive(Debug, Clone, Default)]
pub struct KillSwitchState {
    /// Is kill switch currently active?
    pub active: bool,
    /// Candle index when kill switch was activated
    pub activated_at_candle: Option<usize>,
    /// Current consecutive losses (resets on win)
    pub consecutive_losses: u32,
    /// Consecutive wins AFTER kill switch was activated (for reset condition)
    pub consecutive_wins_after_activation: u32,
    /// EMA50 slope when kill switch was activated
    pub ema50_slope_at_activation: Option<Decimal>,
    /// ATR when kill switch was activated
    pub atr_at_activation: Option<Decimal>,
}

impl KillSwitchState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Activate kill switch
    pub fn activate(
        &mut self,
        candle_idx: usize,
        ema50_slope: Option<Decimal>,
        atr: Option<Decimal>,
    ) {
        self.active = true;
        self.activated_at_candle = Some(candle_idx);
        self.consecutive_wins_after_activation = 0;
        self.ema50_slope_at_activation = ema50_slope;
        self.atr_at_activation = atr;
    }

    /// Check if kill switch can be reset based on conditions
    /// T11.2: Requires: min duration passed, 2+ consecutive wins, improved conditions
    pub fn can_reset(
        &self,
        current_candle: usize,
        min_duration: u32,
        required_wins: u32,
        current_ema50_slope: Option<Decimal>,
        current_atr: Option<Decimal>,
        median_atr: Option<Decimal>,
    ) -> bool {
        if !self.active {
            return false; // Nothing to reset
        }

        // Condition 1: Minimum duration must have passed
        let duration_passed = match self.activated_at_candle {
            Some(activated) => (current_candle.saturating_sub(activated)) as u32 >= min_duration,
            None => false,
        };

        // Condition 2: Required consecutive wins after activation
        let wins_met = self.consecutive_wins_after_activation >= required_wins;

        // Condition 3: EMA50 slope must be positive (trending)
        let slope_positive = current_ema50_slope
            .map(|s| s > Decimal::ZERO)
            .unwrap_or(false);

        // Condition 4: ATR > median ATR (volatility returned)
        let atr_ok = match (current_atr, median_atr) {
            (Some(curr), Some(med)) => curr > med,
            _ => true, // If we don't have ATR data, skip this check
        };

        duration_passed && wins_met && slope_positive && atr_ok
    }

    /// Reset kill switch
    pub fn reset(&mut self) {
        self.active = false;
        self.activated_at_candle = None;
        self.consecutive_wins_after_activation = 0;
        self.ema50_slope_at_activation = None;
        self.atr_at_activation = None;
        // Note: consecutive_losses is NOT reset here - only on individual wins
    }

    /// Record trade result
    pub fn record_result(&mut self, is_win: bool) {
        if is_win {
            self.consecutive_losses = 0;
            if self.active {
                self.consecutive_wins_after_activation += 1;
            }
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins_after_activation = 0;
        }
    }
}

impl Default for PositionPoolConfig {
    fn default() -> Self {
        Self {
            max_active_trades_per_symbol_tf: 3, // Max 3 trades per symbol/TF (T8.1 HARD CAP)
            allow_same_direction: true,         // LONG + LONG allowed (continuation)
            allow_hedge: false,                 // LONG + SHORT NOT allowed
            confidence_reduction_per_trade: 0.3, // 30% reduction per additional trade
            min_score_improvement: 10,          // T8.2: Need +10 score to replace weak trade
            max_trade_duration_candles: 14,     // T9.1: Force exit after 14 candles
            be_threshold_candles: 6,            // T9.2: Apply BE after 6 candles
            be_min_profit_r: Decimal::from_str_exact("0.5").unwrap(), // T9.2: Don't apply BE if > 0.5R profit
            partial_tp_enabled: false,                                // T9.3: Disabled by default
            partial_tp_ratio: 0.5,                                    // T9.3: Close 50% at 1R
            kill_switch_consec_losses: 7, // T10.2: Pause after 7 consecutive losses
            kill_switch_min_duration: 20, // T11.1: DEFAULT - use get_kill_switch_duration_for_tf()
            kill_switch_reset_wins: 2,    // T11.2: Need 2 consecutive wins to reset
        }
    }
}

// =============================================================================
// PHASE A: TF-Based Kill Switch Duration
// =============================================================================

/// Get kill switch minimum duration based on timeframe
/// Shorter TFs need shorter reset periods, longer TFs need longer
pub fn get_kill_switch_duration_for_tf(timeframe: &str) -> u32 {
    match timeframe {
        // Lower timeframes: faster reset (more candles per day)
        "1m" => 60,  // 1 hour worth of candles
        "3m" => 40,  // ~2 hours
        "5m" => 30,  // ~2.5 hours (BLOCKED but kept for reference)
        "15m" => 24, // 6 hours (BLOCKED but kept for reference)
        "30m" => 16, // 8 hours

        // Medium timeframes: standard reset
        "1h" => 12, // 12 hours
        "2h" => 10, // 20 hours
        "4h" => 8,  // 32 hours (~1.3 days)

        // Higher timeframes: slower reset (fewer candles)
        "6h" => 6,  // 36 hours
        "8h" => 5,  // 40 hours
        "12h" => 4, // 48 hours (2 days)
        "1d" => 3,  // 3 days (BLOCKED but kept for reference)
        "3d" => 2,  // 6 days
        "1w" => 2,  // 2 weeks

        _ => 20, // Default fallback
    }
}

/// Position pool manager for multi-position trading
#[derive(Debug, Clone, Default)]
pub struct PositionPool {
    /// All trades (open and closed)
    pub trades: Vec<ActiveTrade>,
    /// Configuration
    pub config: PositionPoolConfig,
}

impl PositionPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: PositionPoolConfig) -> Self {
        Self {
            trades: Vec::new(),
            config,
        }
    }

    /// Get all active (open) trades for a symbol/timeframe
    pub fn active_trades(&self, symbol: &str, timeframe: &str) -> Vec<&ActiveTrade> {
        self.trades
            .iter()
            .filter(|t| t.is_open() && t.signal.symbol == symbol && t.signal.timeframe == timeframe)
            .collect()
    }

    /// Get count of active trades for symbol/timeframe
    pub fn active_count(&self, symbol: &str, timeframe: &str) -> usize {
        self.active_trades(symbol, timeframe).len()
    }

    /// Check if we can open a new trade (all guards)
    pub fn can_open_trade(
        &self,
        symbol: &str,
        timeframe: &str,
        direction: &SignalType,
        context_id: &ContextId,
    ) -> (bool, Option<String>) {
        let active = self.active_trades(symbol, timeframe);

        // Guard 1: Max active trades
        if active.len() >= self.config.max_active_trades_per_symbol_tf {
            return (
                false,
                Some(format!(
                    "Max active trades ({}) reached for {} {}",
                    self.config.max_active_trades_per_symbol_tf, symbol, timeframe
                )),
            );
        }

        // Guard 2: Context uniqueness - no duplicate context_id
        for trade in &active {
            if trade.context_id == *context_id {
                return (
                    false,
                    Some(format!(
                        "Context ID {} already has an active trade",
                        context_id
                    )),
                );
            }
        }

        // Guard 3: No hedge (opposite direction)
        if !self.config.allow_hedge {
            for trade in &active {
                if trade.direction != *direction {
                    return (
                        false,
                        Some(format!(
                            "Hedge not allowed: existing {:?} trade, attempted {:?}",
                            trade.direction, direction
                        )),
                    );
                }
            }
        }

        (true, None)
    }

    /// T8.2: Find the weakest trade (lowest context_score) among active trades
    pub fn find_weakest_trade(&self, symbol: &str, timeframe: &str) -> Option<&ActiveTrade> {
        self.active_trades(symbol, timeframe)
            .into_iter()
            .min_by_key(|t| t.context_score)
    }

    /// T8.2: Check if new trade can replace a weaker trade
    pub fn can_replace_weak_trade(
        &self,
        symbol: &str,
        timeframe: &str,
        new_score: i32,
    ) -> Option<String> {
        if let Some(weakest) = self.find_weakest_trade(symbol, timeframe) {
            let score_improvement = new_score - weakest.context_score;
            if score_improvement >= self.config.min_score_improvement {
                return Some(weakest.trade_id.clone());
            }
        }
        None
    }

    /// T8.2: Remove a trade by ID (for replacing weak trades)
    pub fn remove_trade(&mut self, trade_id: &str) -> Option<ActiveTrade> {
        if let Some(pos) = self.trades.iter().position(|t| t.trade_id == trade_id) {
            Some(self.trades.remove(pos))
        } else {
            None
        }
    }

    /// T8.3: Check for trend saturation (slope weakening)
    /// Returns true if new entry should be blocked due to saturation
    pub fn is_trend_saturated(
        &self,
        symbol: &str,
        timeframe: &str,
        current_slope: Decimal,
        direction: &SignalType,
    ) -> bool {
        let active = self.active_trades(symbol, timeframe);
        if active.is_empty() {
            return false;
        }

        // Check if slope is weakening compared to existing trades
        for trade in &active {
            if let Some(entry_slope) = trade.ema50_slope_at_entry {
                match direction {
                    SignalType::LONG => {
                        // For LONG, slope should still be positive and not significantly weaker
                        if current_slope
                            < entry_slope
                                * Decimal::from_str_exact("0.5")
                                    .unwrap_or(Decimal::ONE / Decimal::from(2))
                        {
                            return true; // Slope dropped by more than 50%
                        }
                    }
                    SignalType::SHORT => {
                        // For SHORT, slope should still be negative and not significantly weaker
                        if current_slope
                            > entry_slope
                                * Decimal::from_str_exact("0.5")
                                    .unwrap_or(Decimal::ONE / Decimal::from(2))
                        {
                            return true; // Slope weakened (became less negative)
                        }
                    }
                }
            }
        }

        false
    }

    /// Calculate adjusted confidence based on active trades
    pub fn calculate_adjusted_confidence(
        &self,
        symbol: &str,
        timeframe: &str,
        base_confidence: u8,
    ) -> u8 {
        let active_count = self.active_count(symbol, timeframe);

        if active_count == 0 {
            return base_confidence;
        }

        // Reduce confidence by configured amount per active trade
        // With 2 trades: confidence = base * 0.6
        let multiplier = 1.0 - (active_count as f64 * self.config.confidence_reduction_per_trade);
        let adjusted = (base_confidence as f64 * multiplier.max(0.2)) as u8;
        adjusted.max(1)
    }

    /// Add a new trade to the pool
    pub fn add_trade(&mut self, trade: ActiveTrade) {
        self.trades.push(trade);
    }

    /// Get mutable reference to active trades for exit processing
    pub fn active_trades_mut(&mut self) -> Vec<&mut ActiveTrade> {
        self.trades.iter_mut().filter(|t| t.is_open()).collect()
    }

    /// Get all completed trades
    pub fn completed_trades(&self) -> Vec<&ActiveTrade> {
        self.trades.iter().filter(|t| !t.is_open()).collect()
    }

    /// Get max concurrent trades (for metrics)
    pub fn max_concurrent_trades(&self) -> usize {
        if self.trades.is_empty() {
            return 0;
        }

        // Find the maximum number of overlapping trades
        let mut max_concurrent = 0;

        for trade in &self.trades {
            let start = trade.opened_at_candle;
            let end = trade.exit_candle_idx.unwrap_or(usize::MAX);

            let concurrent = self
                .trades
                .iter()
                .filter(|t| {
                    let t_start = t.opened_at_candle;
                    let t_end = t.exit_candle_idx.unwrap_or(usize::MAX);
                    // Check overlap
                    t_start <= end && t_end >= start
                })
                .count();

            max_concurrent = max_concurrent.max(concurrent);
        }

        max_concurrent
    }

    /// Get average concurrent trades (for metrics)
    pub fn avg_concurrent_trades(&self) -> f64 {
        let completed = self.completed_trades();
        if completed.is_empty() {
            return 0.0;
        }

        let total_overlap: usize = completed
            .iter()
            .map(|t| {
                let start = t.opened_at_candle;
                let end = t.exit_candle_idx.unwrap_or(start);

                completed
                    .iter()
                    .filter(|other| {
                        let o_start = other.opened_at_candle;
                        let o_end = other.exit_candle_idx.unwrap_or(o_start);
                        o_start <= end && o_end >= start && other.trade_id != t.trade_id
                    })
                    .count()
            })
            .sum();

        1.0 + (total_overlap as f64 / (completed.len() as f64 * 2.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub open_time: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    #[serde(default)]
    pub close_time: Option<DateTime<Utc>>,
}

impl Candle {
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    LONG,
    SHORT,
}

// T6.2 — Signal Output Contracts
pub const ENGINE_VERSION: &str = "2.1.0"; // FT3: Updated version

// =============================================================================
// FINAL TASK 3 — Signal Contract (API Readiness)
// =============================================================================

/// FT3: API-ready signal output contract
/// All fields are required for external consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    // Required API fields (FT3)
    pub signal_id: String,      // Unique signal ID
    pub engine_version: String, // Engine version for compatibility
    pub symbol: String,         // e.g., "BTCUSDT"
    #[serde(rename = "tf")]
    pub timeframe: String, // e.g., "1h"
    #[serde(rename = "direction")]
    pub signal: SignalType, // LONG or SHORT
    pub price: Decimal,         // Entry price
    pub confidence: u8,         // 0-100 normalized score
    pub confidence_tier: String, // "high", "medium", "low"
    pub timestamp: DateTime<Utc>, // Signal generation time
    #[serde(rename = "reason")]
    pub reasons: Vec<String>, // ["Liquidity BOS", "EMA trend", ...]
    pub context_id: Option<String>, // FT3: Context identifier
    pub regime_context: Option<RegimeContext>, // Regime info
}

/// FT3: Simplified API output (for external consumers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalApiOutput {
    pub symbol: String,
    pub tf: String,
    pub direction: String,
    pub confidence: f64, // 0.0-1.0 normalized
    pub context_id: String,
    pub engine_version: String,
    pub timestamp: String, // ISO 8601 format
    pub reason: Vec<String>,
}

impl TradeSignal {
    /// FT3: Convert to API output format
    pub fn to_api_output(&self) -> SignalApiOutput {
        SignalApiOutput {
            symbol: self.symbol.clone(),
            tf: self.timeframe.clone(),
            direction: format!("{:?}", self.signal),
            confidence: self.confidence as f64 / 100.0,
            context_id: self.context_id.clone().unwrap_or_default(),
            engine_version: self.engine_version.clone(),
            timestamp: self.timestamp.to_rfc3339(),
            reason: self.reasons.clone(),
        }
    }

    /// FT3: Serialize to JSON for API consumption
    pub fn to_api_json(&self) -> String {
        serde_json::to_string_pretty(&self.to_api_output()).unwrap_or_default()
    }
}

// T5.2 — Regime Context for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeContext {
    pub atr_regime: String,   // "low", "normal", "high"
    pub slope_regime: String, // "flat", "trending_up", "trending_down"
    pub session: String,      // UTC hour bucket: "asia", "london", "ny"
}

impl TradeSignal {
    pub fn new(
        symbol: String,
        timeframe: String,
        signal: SignalType,
        price: Decimal,
        score: i32,
        reasons: Vec<String>,
        regime: Option<RegimeContext>,
    ) -> Self {
        use uuid::Uuid;

        let confidence = score.clamp(0, 100) as u8;
        let confidence_tier = match confidence {
            80..=100 => "high",
            65..=79 => "medium",
            _ => "low",
        }
        .to_string();

        Self {
            signal_id: Uuid::new_v4().to_string(),
            engine_version: ENGINE_VERSION.to_string(),
            symbol,
            timeframe,
            signal,
            price,
            confidence,
            confidence_tier,
            timestamp: Utc::now(),
            reasons,
            context_id: None, // FT3: Set by caller if needed
            regime_context: regime,
        }
    }
}

impl fmt::Display for TradeSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SIGNAL [{}]: {} [{}] {:?} @ {} (Conf: {}% - {})",
            &self.signal_id[..8],
            self.symbol,
            self.timeframe,
            self.signal,
            self.price,
            self.confidence,
            self.confidence_tier
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendState {
    Bullish,
    Bearish,
    Neutral,
}

#[derive(Debug, Clone)]
pub struct MarketStructure {
    pub last_pivot_high: Option<Decimal>,
    pub last_pivot_low: Option<Decimal>,
    pub trend: TrendState,
    pub bos_confirmed: bool,

    // Liquidity-aware BOS tracking
    pub has_equal_highs: bool,             // Likidite havuzu: eşit high'lar
    pub has_equal_lows: bool,              // Likidite havuzu: eşit low'lar
    pub last_bos_displacement: bool,       // Son BOS displacement ile mi oldu?
    pub bos_candle_range: Option<Decimal>, // BOS mumunun range'i
}

impl Default for MarketStructure {
    fn default() -> Self {
        Self {
            last_pivot_high: None,
            last_pivot_low: None,
            trend: TrendState::Neutral,
            bos_confirmed: false,
            has_equal_highs: false,
            has_equal_lows: false,
            last_bos_displacement: false,
            bos_candle_range: None,
        }
    }
}

// WebSocket Message Types
#[derive(Debug, Deserialize)]
pub struct WsStreamMessage {
    pub stream: String,
    pub data: WsKlineEvent,
}

#[derive(Debug, Deserialize)]
pub struct WsKlineEvent {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "k")]
    pub kline: WsKline,
}

#[derive(Debug, Deserialize)]
pub struct WsKline {
    #[serde(rename = "t")]
    pub start_time: i64,
    #[serde(rename = "T")]
    pub end_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "i")]
    pub interval: String,
    #[serde(rename = "o")]
    pub open: String,
    #[serde(rename = "c")]
    pub close: String,
    #[serde(rename = "h")]
    pub high: String,
    #[serde(rename = "l")]
    pub low: String,
    #[serde(rename = "v")]
    pub volume: String,
    #[serde(rename = "x")]
    pub is_closed: bool,
}

impl WsKline {
    pub fn to_candle(&self) -> anyhow::Result<Candle> {
        Ok(Candle {
            open_time: DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(self.start_time).unwrap(),
                Utc,
            ),
            open: Decimal::from_str_exact(&self.open)?,
            high: Decimal::from_str_exact(&self.high)?,
            low: Decimal::from_str_exact(&self.low)?,
            close: Decimal::from_str_exact(&self.close)?,
            volume: Decimal::from_str_exact(&self.volume)?,
            close_time: Some(DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(self.end_time).unwrap(),
                Utc,
            )),
        })
    }
}
