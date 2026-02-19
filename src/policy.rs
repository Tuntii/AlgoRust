use crate::types::ContextId;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingMode {
    Active,
    Shadow,
    Blocked,
}

pub struct TimeframePolicy {
    active_pairs: HashSet<(String, String)>,
    shadow_pairs: HashSet<(String, String)>,
    blocked_pairs: HashSet<(String, String)>,
}

impl TimeframePolicy {
    pub fn new() -> Self {
        let mut active_pairs = HashSet::new();
        let shadow_pairs = HashSet::new();
        let mut blocked_pairs = HashSet::new();

        blocked_pairs.insert(("BTCUSDT".to_string(), "5m".to_string()));
        blocked_pairs.insert(("ETHUSDT".to_string(), "5m".to_string()));
        blocked_pairs.insert(("SOLUSDT".to_string(), "5m".to_string()));
        blocked_pairs.insert(("SOLUSDT".to_string(), "15m".to_string()));
        blocked_pairs.insert(("BTCUSDT".to_string(), "1d".to_string()));
        blocked_pairs.insert(("ETHUSDT".to_string(), "1d".to_string()));
        blocked_pairs.insert(("SOLUSDT".to_string(), "1d".to_string()));

        // 1m-only active set (current live system)
        active_pairs.insert(("BTCUSDT".to_string(), "1m".to_string()));
        active_pairs.insert(("ETHUSDT".to_string(), "1m".to_string()));

        Self {
            active_pairs,
            shadow_pairs,
            blocked_pairs,
        }
    }

    pub fn new_backtest() -> Self {
        Self {
            active_pairs: HashSet::new(),
            shadow_pairs: HashSet::new(),
            blocked_pairs: HashSet::new(),
        }
    }

    fn is_backtest_mode(&self) -> bool {
        self.active_pairs.is_empty()
            && self.shadow_pairs.is_empty()
            && self.blocked_pairs.is_empty()
    }

    pub fn get_mode(&self, symbol: &str, timeframe: &str) -> TradingMode {
        if self.is_backtest_mode() {
            return TradingMode::Active;
        }

        let key = (symbol.to_string(), timeframe.to_string());
        if self.blocked_pairs.contains(&key) {
            TradingMode::Blocked
        } else if self.active_pairs.contains(&key) {
            TradingMode::Active
        } else if self.shadow_pairs.contains(&key) {
            TradingMode::Shadow
        } else {
            TradingMode::Shadow
        }
    }

    pub fn is_allowed(&self, symbol: &str, timeframe: &str) -> bool {
        self.get_mode(symbol, timeframe) == TradingMode::Active
    }

    pub fn can_generate_signal(&self, symbol: &str, timeframe: &str) -> bool {
        matches!(
            self.get_mode(symbol, timeframe),
            TradingMode::Active | TradingMode::Shadow
        )
    }

    pub fn get_block_reason(&self, symbol: &str, timeframe: &str) -> Option<String> {
        match self.get_mode(symbol, timeframe) {
            TradingMode::Blocked => Some(format!(
                "Policy violation: {} {} is blocked",
                symbol, timeframe
            )),
            TradingMode::Shadow => Some(format!(
                "Shadow mode: {} {} (research only, no trades)",
                symbol, timeframe
            )),
            TradingMode::Active => None,
        }
    }
}

impl Default for TimeframePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BootstrapState {
    pub ema200_ready: bool,
    pub pivot_history_ready: bool,
    pub atr_ready: bool,
    pub candle_count: usize,
    pub timeframe: String,
    pub suppression_reason: Option<String>,
}

impl BootstrapState {
    pub fn new() -> Self {
        Self {
            ema200_ready: false,
            pivot_history_ready: false,
            atr_ready: false,
            candle_count: 0,
            timeframe: "1h".to_string(),
            suppression_reason: None,
        }
    }

    pub fn with_timeframe(timeframe: &str) -> Self {
        Self {
            timeframe: timeframe.to_string(),
            ..Self::new()
        }
    }

    pub fn min_candles_for_tf(timeframe: &str) -> usize {
        match timeframe {
            "1m" => 3000,
            "4h" => 1000,
            "1h" => 700,
            "30m" => 450,
            "15m" => 300,
            _ => 300,
        }
    }

    pub fn update(
        &mut self,
        candle_count: usize,
        has_ema200: bool,
        pivot_count: usize,
        has_atr: bool,
    ) {
        self.candle_count = candle_count;
        let min_candles = Self::min_candles_for_tf(&self.timeframe);

        self.ema200_ready = candle_count >= min_candles && has_ema200;
        self.pivot_history_ready = pivot_count >= 2;
        self.atr_ready = candle_count >= 14 && has_atr;

        if !self.is_complete() {
            let mut reasons = Vec::new();
            if !self.ema200_ready {
                reasons.push(format!(
                    "EMA200 not seeded ({}/{} candles)",
                    candle_count, min_candles
                ));
            }
            if !self.pivot_history_ready {
                reasons.push("Insufficient pivot history".to_string());
            }
            if !self.atr_ready {
                reasons.push("ATR not ready".to_string());
            }
            self.suppression_reason = Some(format!("bootstrap_incomplete: {}", reasons.join(", ")));
        } else {
            self.suppression_reason = None;
        }
    }

    pub fn update_with_tf(
        &mut self,
        timeframe: &str,
        candle_count: usize,
        has_ema200: bool,
        pivot_count: usize,
        has_atr: bool,
    ) {
        self.timeframe = timeframe.to_string();
        self.update(candle_count, has_ema200, pivot_count, has_atr);
    }

    pub fn is_complete(&self) -> bool {
        self.ema200_ready && self.pivot_history_ready && self.atr_ready
    }
}

impl Default for BootstrapState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CooldownManager {
    post_close_cooldowns: HashMap<String, usize>,
    context_cooldowns: HashMap<String, usize>,
    open_trades: HashSet<String>,
    backtest_mode: bool,
}

impl CooldownManager {
    pub fn new() -> Self {
        Self {
            post_close_cooldowns: HashMap::new(),
            context_cooldowns: HashMap::new(),
            open_trades: HashSet::new(),
            backtest_mode: false,
        }
    }

    pub fn with_backtest_mode(backtest_mode: bool) -> Self {
        Self {
            backtest_mode,
            ..Self::new()
        }
    }

    fn cooldown_for_timeframe(&self, timeframe: &str) -> usize {
        if self.backtest_mode {
            0
        } else {
            match timeframe {
                "1m" => 1,
                "5m" => 1,
                "15m" => 1,
                "30m" => 1,
                "1h" => 1,
                _ => 1,
            }
        }
    }

    pub fn is_context_on_cooldown(
        &self,
        context_id: &ContextId,
        timeframe: &str,
        current_candle_idx: usize,
    ) -> bool {
        let key = format!("{}:{}", context_id.context_type, context_id.identifier);
        let Some(last_close_idx) = self.context_cooldowns.get(&key) else {
            return false;
        };
        current_candle_idx < *last_close_idx + self.cooldown_for_timeframe(timeframe)
    }

    pub fn record_context_close(&mut self, context_id: &ContextId, candle_idx: usize) {
        let key = format!("{}:{}", context_id.context_type, context_id.identifier);
        self.context_cooldowns.insert(key, candle_idx);
    }

    pub fn is_on_cooldown(&self, key: &str, current_candle_idx: usize) -> bool {
        let Some(last_close_idx) = self.post_close_cooldowns.get(key) else {
            return false;
        };
        current_candle_idx < *last_close_idx + 1
    }

    pub fn has_open_trade(&self, key: &str) -> bool {
        self.open_trades.contains(key)
    }

    pub fn record_signal(&mut self, key: &str, _candle_idx: usize) {
        self.open_trades.insert(key.to_string());
    }

    pub fn record_trade_close(&mut self, key: &str, candle_idx: usize) {
        self.open_trades.remove(key);
        self.post_close_cooldowns
            .insert(key.to_string(), candle_idx);
    }
}

impl Default for CooldownManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PolicyEngine {
    pub timeframe_policy: TimeframePolicy,
    pub cooldown_manager: CooldownManager,
}

impl PolicyEngine {
    pub fn new() -> Self {
        Self {
            timeframe_policy: TimeframePolicy::new(),
            cooldown_manager: CooldownManager::new(),
        }
    }

    pub fn new_backtest_mode() -> Self {
        Self {
            timeframe_policy: TimeframePolicy::new_backtest(),
            cooldown_manager: CooldownManager::with_backtest_mode(true),
        }
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}
