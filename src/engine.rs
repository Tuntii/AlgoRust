use crate::analytics::BlockStats;
use crate::ml_filter::LstmFilter;
use crate::policy::PolicyEngine;
use crate::state::SymbolContext;
use crate::types::{
    get_kill_switch_duration_for_tf, ActiveTrade, ContextId, KillSwitchState, PositionPool,
    SignalType, TradeSignal,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LstmMode {
    Filter,
    LstmOnly,
}

pub struct SignalEngine {
    policy: PolicyEngine,
    pub block_stats: BlockStats,
    /// Multi-position pool for tracking active trades
    pub position_pool: PositionPool,
    /// Multi-position mode enabled
    pub multi_position_enabled: bool,
    /// T11.3: Kill switch states per symbol+timeframe
    pub kill_switch_states: HashMap<String, KillSwitchState>,
    /// Optional LSTM filter (ONNX)
    pub lstm_filter: Option<LstmFilter>,
    /// LSTM gating mode
    pub lstm_mode: LstmMode,
}

impl SignalEngine {
    pub fn new() -> Self {
        Self {
            policy: PolicyEngine::new(),
            block_stats: BlockStats::new(),
            position_pool: PositionPool::new(),
            multi_position_enabled: true, // Enable by default
            kill_switch_states: HashMap::new(),
            lstm_filter: None,
            lstm_mode: LstmMode::Filter,
        }
    }

    /// Create engine configured for backtest mode (shorter cooldowns)
    pub fn new_backtest_mode() -> Self {
        Self {
            policy: PolicyEngine::new_backtest_mode(),
            block_stats: BlockStats::new(),
            position_pool: PositionPool::new(),
            multi_position_enabled: true,
            kill_switch_states: HashMap::new(),
            lstm_filter: None,
            lstm_mode: LstmMode::Filter,
        }
    }

    pub fn set_lstm_filter(&mut self, filter: LstmFilter) {
        self.lstm_filter = Some(filter);
    }

    pub fn set_lstm_mode(&mut self, mode: LstmMode) {
        self.lstm_mode = mode;
    }

    pub fn get_stats(&self) -> &BlockStats {
        &self.block_stats
    }

    /// Get the position pool
    pub fn get_position_pool(&self) -> &PositionPool {
        &self.position_pool
    }

    /// Get mutable position pool
    pub fn get_position_pool_mut(&mut self) -> &mut PositionPool {
        &mut self.position_pool
    }

    /// T1.4 — Record trade open (signal generated, position entered)
    /// This marks that a trade is active for this symbol/tf
    pub fn record_trade_open(&mut self, symbol: &str, timeframe: &str, candle_idx: usize) {
        let context_key = format!("{}_{}", symbol, timeframe);
        // When a signal is generated, we record it as the last signal time
        // But now with new lifecycle, the trade is "open" until it closes
        self.policy
            .cooldown_manager
            .record_signal(&context_key, candle_idx);
    }

    /// T1.5 — Record trade close (TP/SL/BE hit)
    /// This is when cooldown actually starts
    pub fn record_trade_close(&mut self, symbol: &str, timeframe: &str, candle_idx: usize) {
        let context_key = format!("{}_{}", symbol, timeframe);
        self.policy
            .cooldown_manager
            .record_trade_close(&context_key, candle_idx);
    }

    /// Record context-based trade close (MULTI-POSITION)
    pub fn record_context_close(
        &mut self,
        context_id: &ContextId,
        timeframe: &str,
        candle_idx: usize,
    ) {
        self.policy
            .cooldown_manager
            .record_context_close(context_id, candle_idx);
        let _ = timeframe; // Used for cooldown duration calculation in cooldown manager
    }

    /// T11: Record trade result for kill switch tracking (per symbol+TF)
    /// Now uses STICKY kill switch that doesn't auto-reset on profit
    pub fn record_trade_result(
        &mut self,
        symbol: &str,
        timeframe: &str,
        is_win: bool,
        current_candle: usize,
        ema50_slope: Option<Decimal>,
        atr: Option<Decimal>,
    ) {
        let key = format!("{}_{}", symbol, timeframe);
        let state = self
            .kill_switch_states
            .entry(key.clone())
            .or_insert_with(KillSwitchState::new);
        let config = &self.position_pool.config;

        // Record the trade result
        state.record_result(is_win);

        // Check if kill switch should be ACTIVATED
        if !state.active && state.consecutive_losses >= config.kill_switch_consec_losses {
            let tf_duration = get_kill_switch_duration_for_tf(timeframe);
            state.activate(current_candle, ema50_slope, atr);
            self.block_stats.kill_switch_triggered += 1;
            warn!("🔴 KILL SWITCH ACTIVATED for {} - {} consecutive losses (min {} candles before reset)", 
                  key, state.consecutive_losses, tf_duration);
        }
    }

    /// T11.2: Try to reset kill switch if conditions are met
    /// Called every candle to check reset conditions
    pub fn try_reset_kill_switch(
        &mut self,
        symbol: &str,
        timeframe: &str,
        current_candle: usize,
        current_ema50_slope: Option<Decimal>,
        current_atr: Option<Decimal>,
        median_atr: Option<Decimal>,
    ) -> bool {
        let key = format!("{}_{}", symbol, timeframe);
        let config = &self.position_pool.config;
        let tf_duration = get_kill_switch_duration_for_tf(timeframe);

        if let Some(state) = self.kill_switch_states.get_mut(&key) {
            if state.can_reset(
                current_candle,
                tf_duration, // Use TF-based duration instead of config default
                config.kill_switch_reset_wins,
                current_ema50_slope,
                current_atr,
                median_atr,
            ) {
                state.reset();
                warn!("🟢 KILL SWITCH RESET for {} - conditions met ({}+ wins, slope positive, ATR OK)", 
                      key, config.kill_switch_reset_wins);
                return true;
            }
        }
        false
    }

    /// Add a trade to the position pool
    pub fn add_trade_to_pool(&mut self, trade: ActiveTrade) {
        self.position_pool.add_trade(trade);
    }

    pub fn evaluate(&mut self, ctx: &mut SymbolContext) -> Option<TradeSignal> {
        // User preference: generate signals from indicator only.
        // This bypasses policy/bootstrap/LSTM/cooldown/multi-position entry guards at signal stage.
        let indicator_only_mode = true;
        self.block_stats.total_evaluations += 1;

        let last_candle = ctx.candles.back()?.clone();
        let last_close = last_candle.close;
        let candle_count = ctx.candles.len();
        // Use total_candles_processed for cooldown (tracks absolute candle index)
        let absolute_candle_idx = ctx.total_candles_processed;
        let context_key = format!("{}_{}", ctx.symbol, ctx.timeframe);

        if !indicator_only_mode {
            // ============================================================
            // PHASE 0: Foundation Checks
            // ============================================================

            // T0.1 — Timeframe Policy Enforcement
            if !self
                .policy
                .timeframe_policy
                .is_allowed(&ctx.symbol, &ctx.timeframe)
            {
                if let Some(reason) = self
                    .policy
                    .timeframe_policy
                    .get_block_reason(&ctx.symbol, &ctx.timeframe)
                {
                    warn!("🚫 {}", reason);
                }
                self.block_stats.policy_blocked += 1;
                return None;
            }

            // T0.2 — Bootstrap Integrity Gate
            if !ctx.bootstrap.is_complete() {
                if let Some(ref reason) = ctx.bootstrap.suppression_reason {
                    // Log only occasionally to avoid spam
                    if candle_count % 50 == 0 {
                        warn!("⏳ {} {} - {}", ctx.symbol, ctx.timeframe, reason);
                    }
                }
                self.block_stats.bootstrap_incomplete += 1;
                return None;
            }
        }

        // Generate context ID for this potential signal
        let context_id = ctx.generate_context_id();

        // Get current EMA50 slope for T8.3 trend saturation check
        let current_slope = ctx.get_ema50_slope();

        let lstm_only = self.lstm_mode == LstmMode::LstmOnly;
        let long_signal = ctx.pine_buy_signal || ctx.pine_strong_buy;
        let short_signal = ctx.pine_sell_signal || ctx.pine_strong_sell;

        let direction = if lstm_only {
            if ctx.pine_trend == 1 && ctx.pine_kama_long_filter {
                SignalType::LONG
            } else if ctx.pine_trend == -1 && ctx.pine_kama_short_filter {
                SignalType::SHORT
            } else {
                return None;
            }
        } else {
            if !long_signal && !short_signal {
                return None;
            }
            if long_signal && short_signal {
                return None;
            }

            if long_signal {
                SignalType::LONG
            } else {
                SignalType::SHORT
            }
        };

        let mut lstm_score: Option<f32> = None;
        if !indicator_only_mode {
            if let Some(filter) = &self.lstm_filter {
                match filter.score(ctx) {
                    Ok(Some(score)) => {
                        lstm_score = Some(score);
                        if score < filter.threshold() {
                            self.block_stats.lstm_filtered += 1;
                            return None;
                        }
                    }
                    Ok(None) => {
                        self.block_stats.lstm_filtered += 1;
                        return None;
                    }
                    Err(err) => {
                        warn!(
                            "LSTM filter error for {} {}: {}",
                            ctx.symbol, ctx.timeframe, err
                        );
                    }
                }
            } else if lstm_only {
                warn!(
                    "LSTM-only mode enabled but no filter loaded for {} {}",
                    ctx.symbol, ctx.timeframe
                );
                self.block_stats.lstm_filtered += 1;
                return None;
            }
        }

        if !indicator_only_mode {
            // ============================================================
            // MULTI-POSITION GUARDS (TASK 2 + PHASE 8)
            // ============================================================
            if self.multi_position_enabled {
                // Guard 1: Check if we can open a new trade in the position pool
                let (can_open, block_reason) = self.position_pool.can_open_trade(
                    &ctx.symbol,
                    &ctx.timeframe,
                    &direction,
                    &context_id,
                );

                if !can_open {
                    if let Some(reason) = &block_reason {
                        if reason.contains("Max active trades") {
                            self.block_stats.max_trades_reached += 1;
                        } else if reason.contains("Context ID") {
                            self.block_stats.duplicate_context += 1;
                        } else if reason.contains("Hedge not allowed") {
                            self.block_stats.hedge_blocked += 1;
                        }
                    }
                    return None;
                }

                // Guard 2: Context-based cooldown check
                if self.policy.cooldown_manager.is_context_on_cooldown(
                    &context_id,
                    &ctx.timeframe,
                    absolute_candle_idx,
                ) {
                    self.block_stats.context_cooldown_blocks += 1;
                    return None;
                }

                // T8.3: Trend Saturation Guard
                if self.position_pool.is_trend_saturated(
                    &ctx.symbol,
                    &ctx.timeframe,
                    current_slope,
                    &direction,
                ) {
                    self.block_stats.trend_saturation_blocks += 1;
                    return None;
                }
            } else {
                // LEGACY: Single position mode
                let has_open = self.policy.cooldown_manager.has_open_trade(&context_key);
                if has_open {
                    self.block_stats.open_trade_blocks += 1;
                    return None;
                }

                if self
                    .policy
                    .cooldown_manager
                    .is_on_cooldown(&context_key, absolute_candle_idx)
                {
                    self.block_stats.cooldown_blocks += 1;
                    return None;
                }
            }
        }

        let mut reasons = Vec::new();
        if lstm_only {
            reasons.push("LSTM-only: direction from SuperKAMA trend".to_string());
        } else if ctx.pine_strong_buy || ctx.pine_strong_sell {
            reasons.push("SuperKAMA strong signal: close crossed ATR band".to_string());
        } else {
            reasons.push("SuperKAMA signal: close crossed KAMA".to_string());
        }

        let trend_label = match ctx.pine_trend {
            1 => "bullish",
            -1 => "bearish",
            _ => "neutral",
        };
        if let (Some(kama), Some(upper), Some(lower)) = (
            ctx.kama_10.current_value,
            ctx.pine_upper_band,
            ctx.pine_lower_band,
        ) {
            reasons.push(format!(
                "SuperKAMA state: trend={}, KAMA={}, upper={}, lower={}",
                trend_label, kama, upper, lower
            ));
        } else {
            reasons.push(format!("SuperKAMA state: trend={}", trend_label));
        }
        if let Some(score) = lstm_score {
            reasons.push(format!("LSTM score: {:.3}", score));
        }

        let base_confidence: u8 = if lstm_only {
            70
        } else if ctx.pine_strong_buy || ctx.pine_strong_sell {
            85
        } else {
            75
        };
        let adjusted_confidence = if self.multi_position_enabled {
            self.position_pool.calculate_adjusted_confidence(
                &ctx.symbol,
                &ctx.timeframe,
                base_confidence,
            )
        } else {
            base_confidence
        };

        if adjusted_confidence < base_confidence {
            let active_count = self.position_pool.active_count(&ctx.symbol, &ctx.timeframe);
            reasons.push(format!(
                "Confidence reduced: {} active trades ({}% -> {}%)",
                active_count, base_confidence, adjusted_confidence
            ));
        }

        // Store context ID in context for later use
        ctx.current_context_id = Some(context_id.clone());

        // Record cooldown only when guard stack is active.
        if !indicator_only_mode {
            self.policy
                .cooldown_manager
                .record_signal(&context_key, candle_count);
        }
        ctx.last_signal_candle = Some(candle_count);

        // Track signal generation
        self.block_stats.total_signals_generated += 1;

        let mut signal = TradeSignal::new(
            ctx.symbol.clone(),
            ctx.timeframe.clone(),
            direction,
            last_close,
            adjusted_confidence as i32,
            reasons,
            None,
        );

        signal.confidence = adjusted_confidence;
        signal.confidence_tier = match adjusted_confidence {
            80..=100 => "high",
            65..=79 => "medium",
            _ => "low",
        }
        .to_string();

        Some(signal)
    }
}
