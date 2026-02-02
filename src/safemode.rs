// ============================================================================
// SAFE MODE MODULE — Phase 7 Live Readiness
// ============================================================================

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Serialize, Deserialize};
use tracing::{warn, info, error};

use crate::types::Candle;

// =============================================================================
// T7.1 — Safe Mode: Missing Candle Detection & State Inconsistency
// =============================================================================

#[derive(Debug, Clone)]
pub struct SafeMode {
    /// Symbols currently disabled due to issues
    disabled_symbols: HashMap<String, DisableReason>,
    
    /// Last candle time per symbol/timeframe
    last_candle_times: HashMap<String, DateTime<Utc>>,
    
    /// State validation errors per symbol
    state_errors: HashMap<String, Vec<StateError>>,
    
    /// Max errors before auto-disable
    max_errors_before_disable: u32,
    
    /// Cooldown after disable
    disable_cooldown_minutes: i64,
}

#[derive(Debug, Clone)]
pub struct DisableReason {
    pub reason: String,
    pub disabled_at: DateTime<Utc>,
    pub error_count: u32,
    pub auto_reenable_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct StateError {
    pub error_type: StateErrorType,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateErrorType {
    MissingCandle,
    GapInData,
    StaleData,
    IndicatorInconsistency,
    PriceAnomaly,
    VolumeAnomaly,
}

impl Default for SafeMode {
    fn default() -> Self {
        Self {
            disabled_symbols: HashMap::new(),
            last_candle_times: HashMap::new(),
            state_errors: HashMap::new(),
            max_errors_before_disable: 3,
            disable_cooldown_minutes: 30,
        }
    }
}

impl SafeMode {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_config(max_errors: u32, cooldown_minutes: i64) -> Self {
        Self {
            max_errors_before_disable: max_errors,
            disable_cooldown_minutes: cooldown_minutes,
            ..Self::default()
        }
    }
    
    /// Check if symbol is currently disabled
    pub fn is_symbol_disabled(&self, symbol: &str, timeframe: &str) -> bool {
        let key = format!("{}_{}", symbol, timeframe);
        
        if let Some(reason) = self.disabled_symbols.get(&key) {
            // Check if cooldown has passed
            if let Some(reenable_at) = reason.auto_reenable_at {
                if Utc::now() >= reenable_at {
                    return false; // Cooldown passed, will be re-enabled
                }
            }
            return true;
        }
        false
    }
    
    /// Try to re-enable symbols after cooldown
    pub fn check_reenables(&mut self) {
        let now = Utc::now();
        let to_remove: Vec<String> = self.disabled_symbols
            .iter()
            .filter_map(|(key, reason)| {
                if let Some(reenable_at) = reason.auto_reenable_at {
                    if now >= reenable_at {
                        info!("🔓 Re-enabling {} after cooldown", key);
                        return Some(key.clone());
                    }
                }
                None
            })
            .collect();
        
        for key in to_remove {
            self.disabled_symbols.remove(&key);
            self.state_errors.remove(&key);
        }
    }
    
    /// Validate incoming candle data
    pub fn validate_candle(
        &mut self, 
        symbol: &str, 
        timeframe: &str, 
        candle: &Candle
    ) -> Result<(), String> {
        let key = format!("{}_{}", symbol, timeframe);
        
        // Check for missing candles (gaps)
        if let Some(last_time) = self.last_candle_times.get(&key) {
            let expected_gap = Self::expected_candle_gap(timeframe);
            let actual_gap = candle.open_time.signed_duration_since(*last_time);
            
            // Allow 1.5x expected gap for tolerance
            let max_gap = Duration::seconds((expected_gap.num_seconds() as f64 * 1.5) as i64);
            
            if actual_gap > max_gap {
                let error = StateError {
                    error_type: StateErrorType::GapInData,
                    message: format!(
                        "Gap detected: expected {}s, got {}s", 
                        expected_gap.num_seconds(), 
                        actual_gap.num_seconds()
                    ),
                    timestamp: Utc::now(),
                };
                self.record_error(&key, error);
            }
        }
        
        // Validate candle data integrity
        self.validate_candle_integrity(symbol, timeframe, candle)?;
        
        // Update last candle time
        self.last_candle_times.insert(key, candle.open_time);
        
        Ok(())
    }
    
    /// Check candle data integrity
    fn validate_candle_integrity(
        &mut self,
        symbol: &str,
        timeframe: &str,
        candle: &Candle,
    ) -> Result<(), String> {
        let key = format!("{}_{}", symbol, timeframe);
        
        // High should be >= both open and close
        if candle.high < candle.open || candle.high < candle.close {
            let error = StateError {
                error_type: StateErrorType::PriceAnomaly,
                message: "High price below open/close".to_string(),
                timestamp: Utc::now(),
            };
            self.record_error(&key, error);
            return Err("Invalid candle: high < open/close".to_string());
        }
        
        // Low should be <= both open and close
        if candle.low > candle.open || candle.low > candle.close {
            let error = StateError {
                error_type: StateErrorType::PriceAnomaly,
                message: "Low price above open/close".to_string(),
                timestamp: Utc::now(),
            };
            self.record_error(&key, error);
            return Err("Invalid candle: low > open/close".to_string());
        }
        
        // Check for zero/negative values
        if candle.open <= Decimal::ZERO || candle.high <= Decimal::ZERO 
            || candle.low <= Decimal::ZERO || candle.close <= Decimal::ZERO {
            let error = StateError {
                error_type: StateErrorType::PriceAnomaly,
                message: "Zero or negative price detected".to_string(),
                timestamp: Utc::now(),
            };
            self.record_error(&key, error);
            return Err("Invalid candle: zero/negative price".to_string());
        }
        
        // Check for extreme price moves (> 20% in one candle = likely error)
        let price_change = ((candle.high - candle.low) / candle.low).abs();
        if price_change > Decimal::new(20, 2) { // > 20%
            let error = StateError {
                error_type: StateErrorType::PriceAnomaly,
                message: format!("Extreme price move: {}%", price_change * Decimal::new(100, 0)),
                timestamp: Utc::now(),
            };
            self.record_error(&key, error);
            warn!("⚠️ Extreme price move on {}: {}%", key, price_change * Decimal::new(100, 0));
        }
        
        Ok(())
    }
    
    /// Record a state error and potentially disable symbol
    fn record_error(&mut self, key: &str, error: StateError) {
        warn!("🚨 State error on {}: {:?} - {}", key, error.error_type, error.message);
        
        let errors = self.state_errors.entry(key.to_string()).or_default();
        errors.push(error);
        
        // Check if we should disable
        if errors.len() as u32 >= self.max_errors_before_disable {
            self.disable_symbol(key, "Too many state errors");
        }
    }
    
    /// Manually disable a symbol
    pub fn disable_symbol(&mut self, key: &str, reason: &str) {
        let now = Utc::now();
        let reenable_at = now + Duration::minutes(self.disable_cooldown_minutes);
        
        error!("🛑 DISABLING {}: {} (re-enable at {})", key, reason, reenable_at);
        
        let error_count = self.state_errors.get(key).map(|e| e.len() as u32).unwrap_or(0);
        
        self.disabled_symbols.insert(key.to_string(), DisableReason {
            reason: reason.to_string(),
            disabled_at: now,
            error_count,
            auto_reenable_at: Some(reenable_at),
        });
    }
    
    /// Manually re-enable a symbol
    pub fn enable_symbol(&mut self, key: &str) {
        info!("✅ Manually re-enabling {}", key);
        self.disabled_symbols.remove(key);
        self.state_errors.remove(key);
    }
    
    /// Check for stale data (no updates for extended period)
    pub fn check_stale_data(&mut self, symbol: &str, timeframe: &str) -> bool {
        let key = format!("{}_{}", symbol, timeframe);
        
        if let Some(last_time) = self.last_candle_times.get(&key) {
            let expected_gap = Self::expected_candle_gap(timeframe);
            let max_stale = Duration::seconds(expected_gap.num_seconds() * 3); // 3x gap = stale
            
            if Utc::now().signed_duration_since(*last_time) > max_stale {
                let error = StateError {
                    error_type: StateErrorType::StaleData,
                    message: format!("No data for {}s", max_stale.num_seconds()),
                    timestamp: Utc::now(),
                };
                self.record_error(&key, error);
                return true;
            }
        }
        false
    }
    
    /// Get expected gap between candles
    fn expected_candle_gap(timeframe: &str) -> Duration {
        match timeframe {
            "1m" => Duration::minutes(1),
            "3m" => Duration::minutes(3),
            "5m" => Duration::minutes(5),
            "15m" => Duration::minutes(15),
            "30m" => Duration::minutes(30),
            "1h" => Duration::hours(1),
            "2h" => Duration::hours(2),
            "4h" => Duration::hours(4),
            "6h" => Duration::hours(6),
            "8h" => Duration::hours(8),
            "12h" => Duration::hours(12),
            "1d" => Duration::days(1),
            "3d" => Duration::days(3),
            "1w" => Duration::weeks(1),
            _ => Duration::hours(1), // Default
        }
    }
    
    /// Get status summary
    pub fn status_summary(&self) -> SafeModeStatus {
        SafeModeStatus {
            disabled_count: self.disabled_symbols.len(),
            disabled_symbols: self.disabled_symbols.keys().cloned().collect(),
            total_errors: self.state_errors.values().map(|e| e.len()).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeModeStatus {
    pub disabled_count: usize,
    pub disabled_symbols: Vec<String>,
    pub total_errors: usize,
}

// =============================================================================
// T7.2 — Canary Deployment: Paper Signal Mode
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentMode {
    /// Full production: signals are actionable
    Production,
    
    /// Paper mode: signals logged but not executed
    Paper,
    
    /// Shadow mode: evaluation runs but no output at all
    Shadow,
}

#[derive(Debug, Clone)]
pub struct CanaryDeployment {
    pub mode: DeploymentMode,
    pub paper_signals: Vec<PaperSignal>,
    pub shadow_evaluations: u32,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PaperSignal {
    pub signal_id: String,
    pub symbol: String,
    pub timeframe: String,
    pub direction: String,
    pub entry_price: Decimal,
    pub confidence: u8,
    pub timestamp: DateTime<Utc>,
    pub would_have_outcome: Option<PaperOutcome>,
}

#[derive(Debug, Clone)]
pub struct PaperOutcome {
    pub hit_tp: bool,
    pub hit_sl: bool,
    pub pnl_r: f64,
    pub duration_candles: u32,
}

impl Default for CanaryDeployment {
    fn default() -> Self {
        Self {
            mode: DeploymentMode::Production,
            paper_signals: Vec::new(),
            shadow_evaluations: 0,
            started_at: Utc::now(),
        }
    }
}

impl CanaryDeployment {
    pub fn new(mode: DeploymentMode) -> Self {
        Self {
            mode,
            started_at: Utc::now(),
            ..Self::default()
        }
    }
    
    /// Check if we should output signals
    pub fn should_output_signal(&self) -> bool {
        self.mode == DeploymentMode::Production
    }
    
    /// Check if we should log (paper mode)
    pub fn should_log_signal(&self) -> bool {
        self.mode == DeploymentMode::Production || self.mode == DeploymentMode::Paper
    }
    
    /// Record a paper signal (for paper/shadow modes)
    pub fn record_paper_signal(
        &mut self,
        signal_id: String,
        symbol: String,
        timeframe: String,
        direction: String,
        entry_price: Decimal,
        confidence: u8,
    ) {
        match self.mode {
            DeploymentMode::Paper => {
                info!(
                    "📝 PAPER SIGNAL: {} {} {} @ {} (conf: {}%)",
                    symbol, timeframe, direction, entry_price, confidence
                );
                
                self.paper_signals.push(PaperSignal {
                    signal_id,
                    symbol,
                    timeframe,
                    direction,
                    entry_price,
                    confidence,
                    timestamp: Utc::now(),
                    would_have_outcome: None,
                });
            }
            DeploymentMode::Shadow => {
                self.shadow_evaluations += 1;
                // No logging in shadow mode
            }
            DeploymentMode::Production => {
                // Production mode doesn't record paper signals
            }
        }
    }
    
    /// Update paper signal with outcome (for tracking)
    pub fn update_paper_outcome(
        &mut self,
        signal_id: &str,
        hit_tp: bool,
        hit_sl: bool,
        pnl_r: f64,
        duration_candles: u32,
    ) {
        if let Some(signal) = self.paper_signals.iter_mut()
            .find(|s| s.signal_id == signal_id) 
        {
            signal.would_have_outcome = Some(PaperOutcome {
                hit_tp,
                hit_sl,
                pnl_r,
                duration_candles,
            });
        }
    }
    
    /// Get paper trading summary
    pub fn paper_summary(&self) -> PaperSummary {
        let completed: Vec<&PaperSignal> = self.paper_signals
            .iter()
            .filter(|s| s.would_have_outcome.is_some())
            .collect();
        
        let wins = completed.iter()
            .filter(|s| s.would_have_outcome.as_ref().map(|o| o.hit_tp).unwrap_or(false))
            .count();
        
        let total_r: f64 = completed.iter()
            .filter_map(|s| s.would_have_outcome.as_ref().map(|o| o.pnl_r))
            .sum();
        
        PaperSummary {
            mode: self.mode,
            total_signals: self.paper_signals.len(),
            completed_signals: completed.len(),
            wins,
            losses: completed.len() - wins,
            total_r,
            shadow_evaluations: self.shadow_evaluations,
            running_since: self.started_at,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PaperSummary {
    pub mode: DeploymentMode,
    pub total_signals: usize,
    pub completed_signals: usize,
    pub wins: usize,
    pub losses: usize,
    pub total_r: f64,
    pub shadow_evaluations: u32,
    pub running_since: DateTime<Utc>,
}

impl std::fmt::Display for PaperSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Canary Mode: {:?} | Signals: {} | Completed: {} ({}/{}) | R: {:.2}",
            self.mode,
            self.total_signals,
            self.completed_signals,
            self.wins,
            self.losses,
            self.total_r
        )
    }
}

// =============================================================================
// FINAL TASK 2 — Live Safe Mode Guard
// =============================================================================

/// FT2: Live safety guard that blocks signals when data integrity is compromised
#[derive(Debug, Clone, Default)]
pub struct LiveSafeGuard {
    /// Last candle timestamp per symbol+TF
    last_candle_time: std::collections::HashMap<String, DateTime<Utc>>,
    /// Missing candle count per symbol+TF
    missing_candle_count: std::collections::HashMap<String, u32>,
    /// WS lag detected per symbol+TF  
    ws_lag_detected: std::collections::HashMap<String, bool>,
    /// EMA seed status per symbol+TF
    ema_seeded: std::collections::HashMap<String, bool>,
    /// Max allowed missing candles before blocking
    max_missing_candles: u32,
    /// Max WS lag in seconds before blocking
    max_ws_lag_seconds: i64,
}

impl LiveSafeGuard {
    pub fn new() -> Self {
        Self {
            last_candle_time: std::collections::HashMap::new(),
            missing_candle_count: std::collections::HashMap::new(),
            ws_lag_detected: std::collections::HashMap::new(),
            ema_seeded: std::collections::HashMap::new(),
            max_missing_candles: 2,
            max_ws_lag_seconds: 10,
        }
    }
    
    /// FT2: Check if signal generation is safe
    /// Returns (is_safe, reason) - if not safe, reason explains why
    pub fn is_signal_safe(&self, symbol: &str, timeframe: &str) -> (bool, Option<String>) {
        let key = format!("{}_{}", symbol, timeframe);
        
        // Check 1: EMA seed status
        if !self.ema_seeded.get(&key).copied().unwrap_or(false) {
            return (false, Some("EMA indicators not seeded - NO SIGNAL".to_string()));
        }
        
        // Check 2: Missing candles
        let missing = self.missing_candle_count.get(&key).copied().unwrap_or(0);
        if missing > self.max_missing_candles {
            return (false, Some(format!("Missing {} candles - NO SIGNAL", missing)));
        }
        
        // Check 3: WS lag
        if self.ws_lag_detected.get(&key).copied().unwrap_or(false) {
            return (false, Some("WebSocket lag detected - NO SIGNAL".to_string()));
        }
        
        (true, None)
    }
    
    /// Update candle arrival and check for gaps
    pub fn record_candle(&mut self, symbol: &str, timeframe: &str, candle_time: DateTime<Utc>) {
        let key = format!("{}_{}", symbol, timeframe);
        
        if let Some(last_time) = self.last_candle_time.get(&key) {
            let expected_gap = Self::expected_gap_seconds(timeframe);
            let actual_gap = candle_time.signed_duration_since(*last_time).num_seconds();
            
            // Check for missing candles (gap > 1.5x expected)
            if actual_gap > (expected_gap as f64 * 1.5) as i64 {
                let missing = ((actual_gap / expected_gap) - 1).max(0) as u32;
                *self.missing_candle_count.entry(key.clone()).or_insert(0) += missing;
                warn!("⚠️ Missing {} candle(s) on {} (gap: {}s, expected: {}s)", 
                      missing, key, actual_gap, expected_gap);
            } else {
                // Reset missing count on successful candle
                self.missing_candle_count.insert(key.clone(), 0);
            }
        }
        
        self.last_candle_time.insert(key, candle_time);
    }
    
    /// Check for WS lag (called on each WS message)
    pub fn check_ws_lag(&mut self, symbol: &str, timeframe: &str, server_time: DateTime<Utc>) {
        let key = format!("{}_{}", symbol, timeframe);
        let now = Utc::now();
        let lag = now.signed_duration_since(server_time).num_seconds().abs();
        
        if lag > self.max_ws_lag_seconds {
            warn!("⚠️ WS lag detected on {}: {}s", key, lag);
            self.ws_lag_detected.insert(key, true);
        } else {
            self.ws_lag_detected.insert(key, false);
        }
    }
    
    /// Update EMA seed status
    pub fn set_ema_seeded(&mut self, symbol: &str, timeframe: &str, seeded: bool) {
        let key = format!("{}_{}", symbol, timeframe);
        self.ema_seeded.insert(key, seeded);
    }
    
    /// Reset WS lag flag (called after recovery)
    pub fn reset_ws_lag(&mut self, symbol: &str, timeframe: &str) {
        let key = format!("{}_{}", symbol, timeframe);
        self.ws_lag_detected.insert(key, false);
    }
    
    fn expected_gap_seconds(timeframe: &str) -> i64 {
        match timeframe {
            "1m" => 60,
            "3m" => 180,
            "5m" => 300,
            "15m" => 900,
            "30m" => 1800,
            "1h" => 3600,
            "2h" => 7200,
            "4h" => 14400,
            "6h" => 21600,
            "8h" => 28800,
            "12h" => 43200,
            "1d" => 86400,
            _ => 3600,
        }
    }
}

// =============================================================================
// FINAL TASK 4 — Shadow Live Mode (2 weeks comparison)
// =============================================================================

/// FT4: Shadow signal for live/backtest comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowSignal {
    pub signal_id: String,
    pub symbol: String,
    pub timeframe: String,
    pub direction: String,
    pub entry_price: Decimal,
    pub sl_price: Decimal,
    pub tp_price: Decimal,
    pub confidence: u8,
    pub context_id: String,
    pub reasons: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub candle_idx: usize,
    /// Outcome tracking (filled after resolution)
    pub outcome: Option<String>,
    pub exit_price: Option<Decimal>,
    pub pnl_r: Option<Decimal>,
}

/// FT4: Shadow mode tracker for live/backtest signal comparison
#[derive(Debug, Clone, Default)]
pub struct ShadowLiveTracker {
    /// All shadow signals generated
    pub signals: Vec<ShadowSignal>,
    /// Start time of shadow period
    pub started_at: Option<DateTime<Utc>>,
    /// Whether shadow mode is active
    pub is_active: bool,
    /// Shadow period duration (default 2 weeks)
    pub duration_days: i64,
}

impl ShadowLiveTracker {
    pub fn new() -> Self {
        Self {
            signals: Vec::new(),
            started_at: None,
            is_active: false,
            duration_days: 14, // 2 weeks
        }
    }
    
    /// Start shadow mode
    pub fn start(&mut self) {
        self.started_at = Some(Utc::now());
        self.is_active = true;
        info!("🌑 Shadow Live Mode STARTED - {} days comparison period", self.duration_days);
    }
    
    /// Check if shadow period has ended
    pub fn is_period_complete(&self) -> bool {
        if let Some(started) = self.started_at {
            let elapsed = Utc::now().signed_duration_since(started).num_days();
            elapsed >= self.duration_days
        } else {
            false
        }
    }
    
    /// Record a shadow signal (no actual trade)
    pub fn record_signal(&mut self, signal: ShadowSignal) {
        if self.is_active {
            info!("🌑 SHADOW SIGNAL: {} {} {:?} @ {} (conf: {})", 
                  signal.symbol, signal.timeframe, signal.direction, 
                  signal.entry_price, signal.confidence);
            self.signals.push(signal);
        }
    }
    
    /// Update signal outcome (when SL/TP would have been hit)
    pub fn update_outcome(&mut self, signal_id: &str, outcome: &str, exit_price: Decimal, pnl_r: Decimal) {
        if let Some(sig) = self.signals.iter_mut().find(|s| s.signal_id == signal_id) {
            sig.outcome = Some(outcome.to_string());
            sig.exit_price = Some(exit_price);
            sig.pnl_r = Some(pnl_r);
        }
    }
    
    /// Get shadow mode summary
    pub fn summary(&self) -> ShadowSummary {
        let completed: Vec<_> = self.signals.iter()
            .filter(|s| s.outcome.is_some())
            .collect();
        
        let wins = completed.iter().filter(|s| s.outcome.as_deref() == Some("WIN")).count();
        let losses = completed.iter().filter(|s| s.outcome.as_deref() == Some("LOSS")).count();
        let be = completed.iter().filter(|s| s.outcome.as_deref() == Some("BE")).count();
        
        let total_pnl: Decimal = completed.iter()
            .filter_map(|s| s.pnl_r)
            .sum();
        
        ShadowSummary {
            total_signals: self.signals.len(),
            completed: completed.len(),
            wins,
            losses,
            be,
            total_pnl_r: total_pnl.to_f64().unwrap_or(0.0),
            started_at: self.started_at,
            is_active: self.is_active,
        }
    }
    
    /// Export signals to JSON for comparison
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.signals).unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct ShadowSummary {
    pub total_signals: usize,
    pub completed: usize,
    pub wins: usize,
    pub losses: usize,
    pub be: usize,
    pub total_pnl_r: f64,
    pub started_at: Option<DateTime<Utc>>,
    pub is_active: bool,
}

impl std::fmt::Display for ShadowSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "🌑 Shadow Mode: {} | Signals: {} | Completed: {} (W:{}/L:{}/BE:{}) | PnL: {:.2}R",
            if self.is_active { "ACTIVE" } else { "INACTIVE" },
            self.total_signals,
            self.completed,
            self.wins,
            self.losses,
            self.be,
            self.total_pnl_r
        )
    }
}
