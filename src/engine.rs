use crate::analytics::{BlockStats, ScoreThreshold};
use crate::indicators::DivergenceType;
use crate::mtf_analysis::MTFConfluenceAnalyzer;
use crate::policy::PolicyEngine;
use crate::state::SymbolContext;
use crate::types::{
    get_kill_switch_duration_for_tf, ActiveTrade, ContextId, KillSwitchState, PositionPool,
    PositionPoolConfig, RegimeContext, SignalType, TradeSignal, TrendState,
};
use chrono::Utc;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use std::collections::HashMap;
use tracing::{info, warn};

pub struct SignalEngine {
    policy: PolicyEngine,
    pub block_stats: BlockStats,
    /// Multi-position pool for tracking active trades
    pub position_pool: PositionPool,
    /// Multi-position mode enabled
    pub multi_position_enabled: bool,
    /// T11.3: Kill switch states per symbol+timeframe
    pub kill_switch_states: HashMap<String, KillSwitchState>,
    /// MTF Confluence Analyzer for enhanced signal quality
    pub mtf_analyzer: MTFConfluenceAnalyzer,
}

impl SignalEngine {
    pub fn new() -> Self {
        Self {
            policy: PolicyEngine::new(),
            block_stats: BlockStats::new(),
            position_pool: PositionPool::new(),
            multi_position_enabled: true, // Enable by default
            kill_switch_states: HashMap::new(),
            mtf_analyzer: MTFConfluenceAnalyzer::new(),
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
            mtf_analyzer: MTFConfluenceAnalyzer::new(),
        }
    }

    /// Create engine with custom position pool config
    pub fn with_position_config(config: PositionPoolConfig) -> Self {
        Self {
            policy: PolicyEngine::new(),
            block_stats: BlockStats::new(),
            position_pool: PositionPool::with_config(config),
            multi_position_enabled: true,
            kill_switch_states: HashMap::new(),
            mtf_analyzer: MTFConfluenceAnalyzer::new(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.block_stats = BlockStats::new();
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

    /// T11.3: Check if kill switch is active for specific symbol+TF
    pub fn is_kill_switch_active(&self, symbol: &str, timeframe: &str) -> bool {
        let key = format!("{}_{}", symbol, timeframe);
        self.kill_switch_states
            .get(&key)
            .map(|s| s.active)
            .unwrap_or(false)
    }

    /// T11: Get kill switch state for debugging/logging
    pub fn get_kill_switch_state(&self, symbol: &str, timeframe: &str) -> Option<&KillSwitchState> {
        let key = format!("{}_{}", symbol, timeframe);
        self.kill_switch_states.get(&key)
    }

    /// T11: Manually reset kill switch (for testing)
    pub fn force_reset_kill_switch(&mut self, symbol: &str, timeframe: &str) {
        let key = format!("{}_{}", symbol, timeframe);
        if let Some(state) = self.kill_switch_states.get_mut(&key) {
            state.reset();
            state.consecutive_losses = 0;
        }
    }

    /// Add a trade to the position pool
    pub fn add_trade_to_pool(&mut self, trade: ActiveTrade) {
        self.position_pool.add_trade(trade);
    }

    pub fn evaluate(&mut self, ctx: &mut SymbolContext) -> Option<TradeSignal> {
        self.block_stats.total_evaluations += 1;

        let mut reasons = Vec::new();
        let mut score: i32 = 0;
        let mut signal_type = None;

        let last_candle = ctx.candles.back()?.clone();
        let last_close = last_candle.close;
        let candle_count = ctx.candles.len();
        // Use total_candles_processed for cooldown (tracks absolute candle index)
        let absolute_candle_idx = ctx.total_candles_processed;
        let context_key = format!("{}_{}", ctx.symbol, ctx.timeframe);

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

        // Generate context ID for this potential signal
        let context_id = ctx.generate_context_id();

        // Get current EMA50 slope for T8.3 trend saturation check
        let current_slope = ctx.get_ema50_slope();

        // Determine direction from current trend
        let direction = match ctx.structure.trend {
            TrendState::Bullish => SignalType::LONG,
            TrendState::Bearish => SignalType::SHORT,
            TrendState::Neutral => SignalType::LONG, // Default
        };

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
            // Block new entries if slope is weakening compared to existing trades
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
            // STEP 1: Block if there's an open trade (signal sent, not yet closed)
            let has_open = self.policy.cooldown_manager.has_open_trade(&context_key);
            if has_open {
                self.block_stats.open_trade_blocks += 1;
                return None;
            }

            // STEP 2: Check actual cooldown (only after trade close)
            if self
                .policy
                .cooldown_manager
                .is_on_cooldown(&context_key, absolute_candle_idx)
            {
                self.block_stats.cooldown_blocks += 1;
                return None;
            }
        }

        // ============================================================
        // PHASE 1: Regime & Quality Filters
        // ============================================================

        let current_atr = ctx.atr_14.current_value.unwrap_or_default();
        let atr_ratio = if !last_close.is_zero() {
            current_atr / last_close
        } else {
            Decimal::ZERO
        };
        let median_ratio = ctx.get_median_atr_ratio();
        let slope = ctx.get_ema50_slope();

        // T1.1 — Volatility Regime Filter
        let (vol_penalty, vol_hard_block, vol_reason) =
            self.policy
                .volatility_filter
                .evaluate(&ctx.timeframe, atr_ratio, median_ratio);

        if vol_hard_block {
            // Stats tracked, but no log spam
            self.block_stats.low_atr_blocks += 1;
            return None;
        }

        if vol_penalty != 0 {
            score += vol_penalty;
            if let Some(reason) = vol_reason {
                reasons.push(reason);
            }
        }

        // T1.2 — EMA Slope Filter
        let is_low_atr = vol_penalty < 0;
        let (slope_penalty, slope_hard_block, slope_reason) =
            self.policy.slope_filter.evaluate(slope, is_low_atr);

        if slope_hard_block {
            // Stats tracked, but no log spam
            self.block_stats.flat_ema_blocks += 1;
            return None;
        }

        if slope_penalty != 0 {
            score += slope_penalty;
            if let Some(reason) = slope_reason {
                reasons.push(reason);
            }
        }

        // ============================================================
        // PHASE 3: Asset-Specific Filters
        // ============================================================

        // T3.1 — Wick Trap Filter (SOL-focused)
        let (wick_hard_block, _wick_reason) = self
            .policy
            .wick_trap_filter
            .evaluate(&ctx.symbol, &last_candle);

        if wick_hard_block {
            // Stats tracked, but no log spam - use trace! for debug if needed
            self.block_stats.wick_trap_blocks += 1;
            return None;
        }

        // ============================================================
        // PHASE 2: Structure Intelligence
        // ============================================================

        // === POSITIVE SCORING (TREND DIRECTION) ===
        match ctx.structure.trend {
            TrendState::Bullish => {
                // HTF/LTF conflict check: Slope negative while trend bullish
                if slope < Decimal::ZERO {
                    score -= 20;
                    reasons.push(
                        "⚠️ Trend/Slope conflict: Bullish trend but negative slope (-20)"
                            .to_string(),
                    );
                }

                if ctx.just_confirmed_pivot_low {
                    reasons.push("✅ Bullish market structure confirmed".to_string());
                    reasons.push("✅ Fractal HL detected (Pivot Low confirmed)".to_string());
                    reasons.push("✅ EMA Alignment: 5>8>13>50>200".to_string());

                    score += 30; // Structure
                    score += 25; // EMA
                    score += 20; // Pivot

                    // T2.1 — Liquidity-aware BOS
                    let (bos_adj, _is_strong, bos_reason) = self.policy.liquidity_bos.evaluate(
                        ctx.structure.bos_confirmed,
                        ctx.structure.has_equal_highs,
                        ctx.structure.has_equal_lows,
                        ctx.structure.last_bos_displacement,
                        true, // bullish
                    );
                    score += bos_adj;
                    if let Some(reason) = bos_reason {
                        reasons.push(reason);
                    }

                    // T2.2 — Pivot + Displacement Validation
                    let (disp_adj, disp_reason) = self
                        .policy
                        .displacement_validator
                        .evaluate(&last_candle, true);
                    score += disp_adj;
                    if let Some(reason) = disp_reason {
                        reasons.push(reason);
                    }

                    if let (Some(e50), Some(e13)) =
                        (ctx.ema_50.current_value, ctx.ema_13.current_value)
                    {
                        if last_close > e50 && last_close > e13 {
                            score += 10;
                            reasons.push("✅ Price above EMA13 & EMA50 (+10)".to_string());
                        }
                    }

                    // T3.2 — ETH Micro-Boost
                    let (eth_bonus, eth_reason) = self.policy.eth_micro_boost.evaluate(
                        &ctx.symbol,
                        ctx.structure.has_equal_lows,
                        last_close,
                        ctx.ema_13.current_value,
                        ctx.ema_50.current_value,
                        true,
                    );
                    score += eth_bonus;
                    if let Some(reason) = eth_reason {
                        reasons.push(reason);
                    }

                    // RSI Divergence Boost
                    if ctx.current_divergence == DivergenceType::Bullish {
                        score += 15;
                        reasons.push("🔥 Bullish RSI Divergence Confirmed (+15)".to_string());
                    }

                    signal_type = Some(SignalType::LONG);
                }
            }
            TrendState::Bearish => {
                // HTF/LTF conflict check
                if slope > Decimal::ZERO {
                    score -= 20;
                    reasons.push(
                        "⚠️ Trend/Slope conflict: Bearish trend but positive slope (-20)"
                            .to_string(),
                    );
                }

                if ctx.just_confirmed_pivot_high {
                    reasons.push("✅ Bearish market structure confirmed".to_string());
                    reasons.push("✅ Fractal LH detected (Pivot High confirmed)".to_string());
                    reasons.push("✅ EMA Alignment: 5<8<13<50<200".to_string());

                    score += 30; // Structure
                    score += 25; // EMA
                    score += 20; // Pivot

                    // T2.1 — Liquidity-aware BOS
                    let (bos_adj, _is_strong, bos_reason) = self.policy.liquidity_bos.evaluate(
                        ctx.structure.bos_confirmed,
                        ctx.structure.has_equal_highs,
                        ctx.structure.has_equal_lows,
                        ctx.structure.last_bos_displacement,
                        false, // bearish
                    );
                    score += bos_adj;
                    if let Some(reason) = bos_reason {
                        reasons.push(reason);
                    }

                    // T2.2 — Pivot + Displacement Validation
                    let (disp_adj, disp_reason) = self
                        .policy
                        .displacement_validator
                        .evaluate(&last_candle, false);
                    score += disp_adj;
                    if let Some(reason) = disp_reason {
                        reasons.push(reason);
                    }

                    if let (Some(e50), Some(e13)) =
                        (ctx.ema_50.current_value, ctx.ema_13.current_value)
                    {
                        if last_close < e50 && last_close < e13 {
                            score += 10;
                            reasons.push("✅ Price below EMA13 & EMA50 (+10)".to_string());
                        }
                    }

                    // T3.2 — ETH Micro-Boost
                    let (eth_bonus, eth_reason) = self.policy.eth_micro_boost.evaluate(
                        &ctx.symbol,
                        ctx.structure.has_equal_highs,
                        last_close,
                        ctx.ema_13.current_value,
                        ctx.ema_50.current_value,
                        false,
                    );
                    score += eth_bonus;
                    if let Some(reason) = eth_reason {
                        reasons.push(reason);
                    }

                    // RSI Divergence Boost
                    if ctx.current_divergence == DivergenceType::Bearish {
                        score += 15;
                        reasons.push("🔥 Bearish RSI Divergence Confirmed (+15)".to_string());
                    }

                    signal_type = Some(SignalType::SHORT);
                }
            }
            _ => {}
        }

        // ============================================================
        // FINAL DECISION
        // ============================================================
        if let Some(sig) = signal_type {
            // MTF CONFLUENCE SCORING
            // Calculate confluence score using quick evaluation
            let is_bullish = sig == SignalType::LONG;
            let confluence = self.mtf_analyzer.quick_evaluate(
                last_close,
                ctx.ema_13.current_value,
                ctx.ema_50.current_value,
                ctx.ema_200.current_value,
                ctx.structure.bos_confirmed,
                ctx.structure.last_bos_displacement,
                is_bullish,
                &last_candle,
            );

            // Add confluence score (can be negative for conflicts!)
            score += confluence.score;
            reasons.push(confluence.description.clone());

            // Log confluence for debugging
            if confluence.alignment_count >= 2 {
                info!(
                    "🎯 {} {} - {} (+{})",
                    ctx.symbol, ctx.timeframe, confluence.description, confluence.score
                );
            }

            // T4.2: Use timeframe-based threshold
            let min_score = ScoreThreshold::min_score_for_tf(&ctx.timeframe);

            if score >= min_score {
                // Store context ID in context for later use
                ctx.current_context_id = Some(context_id.clone());

                // Record cooldown
                self.policy
                    .cooldown_manager
                    .record_signal(&context_key, candle_count);
                ctx.last_signal_candle = Some(candle_count);

                // T5.2: Build regime context
                let hour_utc = Utc::now()
                    .format("%H")
                    .to_string()
                    .parse::<u32>()
                    .unwrap_or(12);
                let regime_context =
                    RegimeContext::determine(current_atr, ctx.get_avg_atr(), slope, hour_utc);

                // TASK 5: Risk normalization - adjust confidence based on active trades
                let base_confidence = score.clamp(0, 100) as u8;
                let adjusted_confidence = if self.multi_position_enabled {
                    self.position_pool.calculate_adjusted_confidence(
                        &ctx.symbol,
                        &ctx.timeframe,
                        base_confidence,
                    )
                } else {
                    base_confidence
                };

                // Add confidence adjustment reason if reduced
                if adjusted_confidence < base_confidence {
                    let active_count = self.position_pool.active_count(&ctx.symbol, &ctx.timeframe);
                    reasons.push(format!(
                        "⚠️ Confidence reduced: {} active trades ({}% → {}%)",
                        active_count, base_confidence, adjusted_confidence
                    ));
                }

                // Track signal generation
                self.block_stats.total_signals_generated += 1;

                // T6.1 & T6.2: Use TradeSignal::new for proper ID and versioning
                let mut signal = TradeSignal::new(
                    ctx.symbol.clone(),
                    ctx.timeframe.clone(),
                    sig,
                    last_close,
                    adjusted_confidence as i32, // Use adjusted confidence
                    reasons,
                    Some(regime_context),
                );

                // Override confidence with adjusted value
                signal.confidence = adjusted_confidence;
                signal.confidence_tier = match adjusted_confidence {
                    80..=100 => "high",
                    65..=79 => "medium",
                    _ => "low",
                }
                .to_string();

                return Some(signal);
            } else {
                // Score too low
                self.block_stats.score_too_low += 1;
            }
        }

        None
    }
}
