use crate::analytics::BlockStats;
use crate::ml_filter::LstmFilter;
use crate::policy::PolicyEngine;
use crate::state::SymbolContext;
use crate::types::{
    get_kill_switch_duration_for_tf, ActiveTrade, ContextId, KillSwitchState, PositionPool,
    SignalType, TradeSignal,
};
use rust_decimal::prelude::*;
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
        // All filter guards are active. Set to true only for raw indicator debugging.
        let indicator_only_mode = false;
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

            // ============================================================
            // PHASE 0.3: Market Quality Filters
            // These prevent trading in ranging/compressed/sideways markets
            // which are the primary cause of BE trades.
            // ============================================================

            // Filter 1: Flat EMA — EMA50 slope too flat = sideways market
            // Threshold: |slope| < 0.00005 (0.005% change over 6 bars)
            // 1m context: BTC@95k → blocks only if EMA moved < ~4.75 USDT in 6 minutes
            let ema50_slope = ctx.get_ema50_slope();
            let slope_threshold =
                rust_decimal::Decimal::from_str("0.00005").unwrap_or(rust_decimal::Decimal::ZERO);
            if ema50_slope.abs() < slope_threshold {
                self.block_stats.flat_ema_blocks += 1;
                return None;
            }

            // Filter 2: Low ATR — volatility too compressed = ranging market
            // Skip if current ATR < 50% of median ATR (last 200 bars)
            if let Some(current_atr) = ctx.atr_14.current_value {
                let median_atr_ratio = ctx.get_median_atr_ratio();
                if !median_atr_ratio.is_zero() && !current_atr.is_zero() {
                    // current_atr / close gives ratio; compare to median ratio
                    let current_ratio = if !ctx
                        .candles
                        .back()
                        .map(|c| c.close)
                        .unwrap_or_default()
                        .is_zero()
                    {
                        current_atr / ctx.candles.back().unwrap().close
                    } else {
                        rust_decimal::Decimal::ZERO
                    };
                    let atr_ratio_threshold = median_atr_ratio
                        * rust_decimal::Decimal::from_str("0.5")
                            .unwrap_or(rust_decimal::Decimal::ONE);
                    if current_ratio < atr_ratio_threshold {
                        self.block_stats.low_atr_blocks += 1;
                        return None;
                    }
                }
            }

            // Filter 3: Neutral Trend — EMA stack not aligned = no clear direction
            // Requires EMA5 > EMA8 > EMA13 > EMA50 (bull) or reverse (bear)
            use crate::types::TrendState;
            if ctx.structure.trend == TrendState::Neutral {
                self.block_stats.neutral_trend_blocks += 1;
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
        let is_strong_signal = ctx.pine_strong_buy || ctx.pine_strong_sell;

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

        // ============================================================
        // WHIPSAW QUALITY GATE — Multi-Factor Signal Quality Filter
        // Distinguishes noisy KAMA crossovers (ranging) from real
        // breakouts/breakdowns. NOT time-based — big moves pass instantly.
        // ============================================================
        if !indicator_only_mode && !lstm_only {
            let quality = self.compute_signal_quality(ctx, is_strong_signal);
            let min_quality = 2i32;

            if quality < min_quality {
                self.block_stats.whipsaw_filtered += 1;
                return None;
            }
        }

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

    /// Compute multi-factor signal quality score.
    /// Returns 0..10+. Higher = stronger signal. Minimum 3 required to pass.
    ///
    /// Strong signals (ATR band crosses) and volume spikes bypass the gate
    /// so we NEVER miss a big market move on 1m scalping.
    fn compute_signal_quality(&self, ctx: &SymbolContext, is_strong_signal: bool) -> i32 {
        // ── Override: strong signals always pass ──────────────────────
        if is_strong_signal {
            return 10; // ATR band cross = high volatility move, always trade
        }

        let mut quality = 0i32;

        // ── Factor 1: Short-term Efficiency Ratio (10 bar) ───────────
        // KAMA(2584) ER is almost always ~0 on 1m. Use a 10-bar ER instead
        // to measure recent price efficiency around the signal candle.
        // ER ~ 0 = choppy noise, ER > 0.3 = clear trend
        let er_short = if ctx.candles.len() >= 11 {
            let n = 10usize;
            let last_idx = ctx.candles.len() - 1;
            let first_idx = last_idx - n;
            let direction = (ctx.candles[last_idx].close - ctx.candles[first_idx].close).abs();
            let volatility: Decimal = (first_idx..last_idx)
                .map(|i| (ctx.candles[i + 1].close - ctx.candles[i].close).abs())
                .sum();
            if volatility.is_zero() { Decimal::ZERO } else { direction / volatility }
        } else {
            Decimal::ZERO
        };
        let er_010 = Decimal::from_str("0.10").unwrap_or(Decimal::ZERO);
        let er_025 = Decimal::from_str("0.25").unwrap_or(Decimal::ZERO);
        if er_short > er_025 {
            quality += 3;
        } else if er_short > er_010 {
            quality += 1;
        }

        // ── Factor 2: ADX (directional strength) ─────────────────────
        let adx = ctx.adx_10_10.adx.unwrap_or(Decimal::ZERO);
        let adx_20 = Decimal::from(20);
        let adx_30 = Decimal::from(30);
        if adx > adx_30 {
            quality += 2;
        } else if adx > adx_20 {
            quality += 1;
        }

        // ── Factor 3: Volume confirmation ────────────────────────────
        // Compare last candle volume to 20-bar average
        if ctx.candles.len() >= 20 {
            if let Some(last_candle) = ctx.candles.back() {
                let vol_sum: Decimal = ctx
                    .candles
                    .iter()
                    .rev()
                    .take(20)
                    .map(|c| c.volume)
                    .sum();
                let vol_avg = vol_sum / Decimal::from(20);
                if !vol_avg.is_zero() {
                    let vol_ratio = last_candle.volume / vol_avg;
                    let vol_3x = Decimal::from(3);
                    let vol_13 = Decimal::from_str("1.3").unwrap_or(Decimal::ONE);

                    // Override: extreme volume spike = big move, bypass gate entirely
                    if vol_ratio >= vol_3x {
                        return 10;
                    }
                    if vol_ratio > vol_13 {
                        quality += 1;
                    }
                }
            }
        }

        // ── Factor 4: Candle body strength ───────────────────────────
        // |close - open| / (high - low). Big body = conviction, small = indecision
        if let Some(last) = ctx.candles.back() {
            let range = last.high - last.low;
            let min_range = Decimal::from_str("0.01").unwrap_or(Decimal::ONE);
            if range > min_range {
                let body = (last.close - last.open).abs();
                let body_ratio = body / range;
                let threshold = Decimal::from_str("0.6").unwrap_or(Decimal::ONE);
                if body_ratio > threshold {
                    quality += 1;
                }
            }
        }

        // ── Factor 5: KAMA slope normalised ──────────────────────────
        // If KAMA is actually moving relative to ATR
        if let Some(slope) = ctx.pine_kama_slope_norm {
            let slope_min = Decimal::from_str("0.002").unwrap_or(Decimal::ZERO);
            if slope.abs() > slope_min {
                quality += 1;
            }
        }

        // ── Factor 6: ATR expansion (volatility growing) ────────────
        // current superkama_atr vs average of last 5
        if ctx.superkama_atr_series.len() >= 6 {
            let current_atr = ctx
                .superkama_atr_series
                .back()
                .copied()
                .unwrap_or(Decimal::ZERO);
            let prev_sum: Decimal = ctx
                .superkama_atr_series
                .iter()
                .rev()
                .skip(1)
                .take(5)
                .copied()
                .sum();
            let prev_avg = prev_sum / Decimal::from(5);
            if !prev_avg.is_zero() {
                let expansion = current_atr / prev_avg;
                let exp_threshold = Decimal::from_str("1.3").unwrap_or(Decimal::ONE);
                if expansion > exp_threshold {
                    quality += 1;
                }
            }
        }

        quality
    }
}
