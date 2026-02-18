use crate::alpaca::{AlpacaClient, OrderRequest, OrderType, Side, TimeInForce}; // Added Alpaca types
use crate::analytics::{AdvancedMetrics, BlockStats, RegimeReport, TradeRecord};
use crate::connect::BinanceClient;
use crate::engine::{LstmMode, SignalEngine};
use crate::ml_filter::LstmFilter;
use crate::policy::TimeframePolicy;
use crate::state::SymbolContext;
use crate::types::{ActiveTrade, Candle, ContextId, SignalType, TradeSignal};
use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitMode {
    Supertrend,
    SlTp,
    Hybrid,
}

impl ExitMode {
    fn from_str(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "supertrend" => ExitMode::Supertrend,
            "hybrid" => ExitMode::Hybrid,
            _ => ExitMode::SlTp,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PoolConfigOverrides {
    pub be_threshold_candles: Option<u32>,
    pub be_min_profit_r: Option<f64>,
}

fn apply_pool_config_overrides(engine: &mut SignalEngine, overrides: Option<PoolConfigOverrides>) {
    let Some(overrides) = overrides else {
        return;
    };

    let pool_config = &mut engine.get_position_pool_mut().config;

    if let Some(v) = overrides.be_threshold_candles {
        pool_config.be_threshold_candles = v;
        info!("T9.2 override: be_threshold_candles={}", v);
    }

    if let Some(v) = overrides.be_min_profit_r {
        if let Some(decimal_v) = Decimal::from_f64(v) {
            pool_config.be_min_profit_r = decimal_v;
            info!("T9.2 override: be_min_profit_r={}", decimal_v);
        } else {
            warn!(
                "Invalid be_min_profit_r override: {} (keeping default {})",
                v, pool_config.be_min_profit_r
            );
        }
    }
}

// =============================================================================
// BACKTEST SUMMARY REPORT
// =============================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct BacktestSummary {
    pub total_symbols: usize,
    pub total_timeframes_tested: usize,
    /// All trades that were opened (including BE and MAX_DURATION exits)
    pub total_opened_trades: usize,
    /// Only WIN + LOSS trades (decisive outcomes)
    pub total_trades: usize,
    pub total_wins: usize,
    pub total_losses: usize,
    pub overall_win_rate: f64,
    pub overall_pnl_r: Decimal,
    pub overall_expectancy: f64,
    pub overall_profit_factor: f64,
    pub best_performer: Option<String>,
    pub worst_performer: Option<String>,
    pub block_stats: BlockStats,
    pub results_by_pair: Vec<PairResult>,
    // Dollar P&L calculations (with $1000 per pair)
    pub dollar_pnl: Option<DollarPnlSummary>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DollarPnlSummary {
    pub starting_capital_per_pair: f64,
    pub total_starting_capital: f64,
    pub risk_scenarios: Vec<RiskScenario>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskScenario {
    pub risk_percent: f64,
    pub risk_per_r: f64,
    pub total_dollar_pnl: f64,
    pub total_final_capital: f64,
    pub total_return_percent: f64,
    pub pair_results: Vec<DollarPairResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DollarPairResult {
    pub pair: String,
    pub starting_capital: f64,
    pub dollar_pnl: f64,
    pub final_capital: f64,
    pub return_percent: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairResult {
    pub symbol: String,
    pub timeframe: String,
    /// Decisive trades (WIN + LOSS only)
    pub trades: usize,
    /// All completed trades (WIN + LOSS + BE + MAX_DURATION)
    pub opened_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub pnl_r: Decimal,
    pub expectancy: f64,
    pub profit_factor: f64,
    pub sharpe: f64,
    pub max_consec_loss: u32,
    /// Gross winning R (for overall PF calculation)
    pub gross_wins_r: f64,
    /// Gross losing R absolute value (for overall PF calculation)
    pub gross_losses_r: f64,
}

// T5: Extended backtest result with advanced metrics
#[derive(Serialize)]
struct BacktestResult {
    symbol: String,
    timeframe: String,
    total_trades: usize,
    wins: usize,
    losses: usize,
    win_rate: f64,
    total_pnl_r: Decimal,
    // T5.1: Advanced Metrics
    advanced_metrics: AdvancedMetrics,
    // T5.2: Regime-Based Reporting
    regime_report: RegimeReport,
    signals: Vec<SimulatedTrade>,
}

#[derive(Serialize, Clone)]
struct SimulatedTrade {
    #[serde(flatten)]
    signal: TradeSignal,
    entry_price: Decimal,
    sl_price: Decimal,
    original_sl_price: Decimal, // T9.2: For BE tracking
    tp_price: Decimal,
    exit_price: Option<Decimal>,
    pnl_r: Option<Decimal>,
    outcome: Option<String>,
    // T5.1: Trade duration tracking
    entry_candle_idx: usize,
    exit_candle_idx: Option<usize>,
    duration_candles: Option<u32>,
    // MULTI-POSITION: Context tracking
    context_id: Option<ContextId>,
    adjusted_confidence: u8,
    was_concurrent: bool, // True if overlapped with another trade
    // T8.2: Context score for ranking
    context_score: i32,
    // T8.3: EMA50 slope at entry
    ema50_slope_at_entry: Option<Decimal>,
    // T9.2: BE tracking
    is_be_applied: bool,
}

pub async fn run_backtest(
    symbols: &[String],
    timeframes: &[String],
    days: i64,
    exit_mode: &str,
    send_alpaca_signals: bool,
    output_dir: &str,
    binance_settings: &crate::connect::BinanceSettings,
    lstm_filter: Option<LstmFilter>,
    lstm_mode: LstmMode,
    pool_overrides: Option<PoolConfigOverrides>,
) -> anyhow::Result<()> {
    info!("🔄 Backtest Başlatılıyor... (Son {} gün)", days);
    info!("═══════════════════════════════════════════════════════════════");

    // Klasörü oluştur
    fs::create_dir_all(output_dir)?;

    let client = BinanceClient::with_settings(binance_settings);
    let mut engine = SignalEngine::new_backtest_mode(); // Backtest mode: bypasses policy
    if let Some(filter) = lstm_filter {
        engine.set_lstm_filter(filter);
    }
    engine.set_lstm_mode(lstm_mode);
    apply_pool_config_overrides(&mut engine, pool_overrides);
    let policy = TimeframePolicy::new();
    let exit_mode = ExitMode::from_str(exit_mode);

    // Alpaca Client (Optional) - .env yüklü ise aktif olur
    let alpaca_client = if send_alpaca_signals && std::env::var("ALPACA_API_KEY").is_ok() {
        match AlpacaClient::new() {
            Ok(c) => {
                info!("🦙 Alpaca entegrasyonu aktif (Backtest Sinyalleri İletilecek)");
                Some(c)
            }
            Err(e) => {
                warn!("⚠️ Alpaca başlatılamadı: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Summary tracking
    let mut summary = BacktestSummary::default();
    let mut all_results: Vec<PairResult> = Vec::new();
    let mut best_pnl = Decimal::MIN;
    let mut worst_pnl = Decimal::MAX;

    // Binance limit: 1000 candles per call.
    // Basitlik için backtest şimdilik son 1000 mum ile sınırlı,
    // ama pagination ile geriye gidilebilir.

    for symbol in symbols {
        for interval in timeframes {
            // T0.1 — Timeframe Policy Enforcement
            // Backtest: Allow both Active AND Shadow mode pairs for testing
            if !policy.can_generate_signal(symbol, interval) {
                warn!("🚫 Skipping {} {} - blocked by policy", symbol, interval);
                continue;
            }

            info!("Testing {} {}...", symbol, interval);

            let mut ctx = SymbolContext::new(symbol.clone(), interval.clone());
            let mut trades: Vec<SimulatedTrade> = Vec::new();
            let mut candle_idx: usize = 0;

            // Veri Çekme (REST)
            // Gerçek bir backtest için pagination gerekir (döngü ile start_time geriye giderek).
            // Şimdilik bootstrap mantığıyla son 1000 mumu test ediyoruz.
            // Modified: Use fetch_historical_candles for full range
            match client
                .fetch_historical_candles(symbol, interval, days)
                .await
            {
                Ok(candles) => {
                    info!("Data loaded: {} candles", candles.len());

                    for candle in candles.iter() {
                        // 1. Mumu ekle
                        ctx.add_candle(candle.clone());

                        // 2. Sinyal kontrol et
                        if let Some(signal) = engine.evaluate(&mut ctx) {
                            // Basit SL/TP Stratejisi
                            // LONG: SL = Last Pivot Low, TP = 1.5R
                            // SHORT: SL = Last Pivot High, TP = 1.5R

                            let entry = candle.close;
                            let (sl, tp) = calculate_sl_tp(&signal, &ctx, entry);

                            // Get context ID from context (set during evaluate)
                            let context_id = ctx.current_context_id.clone();
                            let adjusted_confidence = signal.confidence;

                            // T8.2: Get context score from signal
                            let context_score = signal.confidence as i32;

                            // T8.3: Get EMA50 slope at entry
                            let ema50_slope = ctx.get_ema50_slope();

                            // Check if this trade overlaps with existing trades
                            let active_count =
                                engine.get_position_pool().active_count(symbol, interval);
                            let was_concurrent = active_count > 0;

                            // T1.4: Record trade open
                            engine.record_trade_open(symbol, interval, candle_idx);

                            // Create ActiveTrade for position pool
                            if let Some(ref ctx_id) = context_id {
                                let active_trade = ActiveTrade::new(
                                    signal.clone(),
                                    entry,
                                    sl,
                                    tp,
                                    ctx_id.clone(),
                                    candle_idx,
                                )
                                .with_context_score(context_score)
                                .with_ema50_slope(ema50_slope);
                                engine.add_trade_to_pool(active_trade);

                                // 🦙 Alpaca Entegrasyonu: Backtest sırasında da çalıştırma
                                if let Some(client) = &alpaca_client {
                                    // Sadece son 1 saat içindeki sinyalleri yolla (Koruma)
                                    // VEYA tüm sinyalleri yolla (Kullanıcı isteği - "backtestte yollasın")
                                    // Sorumluluk kullanıcıda. Rate limit yiyebilir.

                                    // Kullanıcı explicit olarak istediği için, zaman kontrolünü es geçiyoruz veya
                                    // basit bir log ile uyarıyoruz.

                                    let side = match signal.signal {
                                        SignalType::LONG => Side::Buy,
                                        SignalType::SHORT => Side::Sell,
                                    };

                                    let order_req = OrderRequest {
                                        symbol: symbol.clone(), // BTCUSDT formatında, Alpaca BTC/USD isteyebilir. Düzenlenmeli.
                                        qty: Decimal::from_str("0.001").unwrap(), // Test Miktarı
                                        side,
                                        order_type: OrderType::Market,
                                        time_in_force: TimeInForce::Gtc,
                                        limit_price: None,
                                        stop_price: None,
                                        order_class: None,
                                        take_profit: None,
                                        stop_loss: None,
                                    };

                                    // Alpaca sembol dönüşümü (Binance -> Alpaca)
                                    // BTCUSDT -> BTC/USD
                                    let mut alpaca_symbol = order_req.symbol.clone();
                                    if alpaca_symbol.ends_with("USDT") {
                                        alpaca_symbol =
                                            format!("{}/USD", alpaca_symbol.replace("USDT", ""));
                                    }
                                    let final_req = OrderRequest {
                                        symbol: alpaca_symbol,
                                        ..order_req
                                    };

                                    match client.submit_order(final_req).await {
                                        Ok(resp) => info!(
                                            "🦙 Alpaca Sinyal İletildi: {} -> ID: {}",
                                            symbol, resp.id
                                        ),
                                        Err(e) => warn!("⚠️ Alpaca Sinyal Hatası: {}", e),
                                    }
                                }
                            }

                            trades.push(SimulatedTrade {
                                signal,
                                entry_price: entry,
                                sl_price: sl,
                                original_sl_price: sl,
                                tp_price: tp,
                                exit_price: None,
                                pnl_r: None,
                                outcome: None,
                                entry_candle_idx: candle_idx,
                                exit_candle_idx: None,
                                duration_candles: None,
                                context_id,
                                adjusted_confidence,
                                was_concurrent,
                                context_score,
                                ema50_slope_at_entry: Some(ema50_slope),
                                is_be_applied: false,
                            });
                        }

                        // 3. Açık pozisyonları yönet (Simülasyon)
                        let pool_config = engine.get_position_pool().config.clone();
                        let allow_supertrend_exit = exit_mode != ExitMode::SlTp;
                        let allow_sl_tp_exit = exit_mode != ExitMode::Supertrend;

                        for trade in trades.iter_mut() {
                            if trade.outcome.is_some() {
                                continue;
                            } // Zaten kapandı

                            // Trade sinyal mumuyla aynıysa skip et
                            if trade.signal.timestamp
                                == candle.close_time.unwrap_or(candle.open_time)
                            {
                                continue;
                            }

                            let mut just_closed = false;
                            let current_duration = (candle_idx - trade.entry_candle_idx) as u32;

                            // T9.1: Max Duration HARD CAP
                            if current_duration >= pool_config.max_trade_duration_candles {
                                let risk = (trade.entry_price - trade.original_sl_price).abs();
                                let unrealized_pnl = match trade.signal.signal {
                                    SignalType::LONG => (candle.close - trade.entry_price) / risk,
                                    SignalType::SHORT => (trade.entry_price - candle.close) / risk,
                                };

                                trade.outcome = Some("MAX_DURATION".to_string());
                                trade.exit_price = Some(candle.close);
                                trade.pnl_r = Some(unrealized_pnl);
                                trade.exit_candle_idx = Some(candle_idx);
                                trade.duration_candles = Some(current_duration);
                                just_closed = true;
                                engine.block_stats.max_duration_exits += 1;
                            }

                            if !just_closed && allow_supertrend_exit {
                                let reversal = match trade.signal.signal {
                                    SignalType::LONG => ctx.pine_trend_changed_bearish,
                                    SignalType::SHORT => ctx.pine_trend_changed_bullish,
                                };

                                if reversal {
                                    let risk = (trade.entry_price - trade.original_sl_price).abs();
                                    let pnl_r = if risk.is_zero() {
                                        Decimal::ZERO
                                    } else {
                                        match trade.signal.signal {
                                            SignalType::LONG => {
                                                (candle.close - trade.entry_price) / risk
                                            }
                                            SignalType::SHORT => {
                                                (trade.entry_price - candle.close) / risk
                                            }
                                        }
                                    };

                                    trade.outcome = Some(
                                        if pnl_r > Decimal::ZERO {
                                            "WIN"
                                        } else if pnl_r < Decimal::ZERO {
                                            "LOSS"
                                        } else {
                                            "BE"
                                        }
                                        .to_string(),
                                    );
                                    trade.exit_price = Some(candle.close);
                                    trade.pnl_r = Some(pnl_r);
                                    trade.exit_candle_idx = Some(candle_idx);
                                    trade.duration_candles = Some(current_duration);
                                    just_closed = true;
                                }
                            }

                            // T9.2: Time-based BE
                            if allow_sl_tp_exit
                                && !just_closed
                                && !trade.is_be_applied
                                && current_duration >= pool_config.be_threshold_candles
                            {
                                let risk = (trade.entry_price - trade.original_sl_price).abs();
                                let unrealized_r = match trade.signal.signal {
                                    SignalType::LONG => (candle.close - trade.entry_price) / risk,
                                    SignalType::SHORT => (trade.entry_price - candle.close) / risk,
                                };

                                if unrealized_r < pool_config.be_min_profit_r {
                                    trade.sl_price = trade.entry_price;
                                    trade.is_be_applied = true;
                                    engine.block_stats.be_applied_count += 1;
                                }
                            }

                            // Normal SL/TP checks
                            if allow_sl_tp_exit && !just_closed {
                                match trade.signal.signal {
                                    SignalType::LONG => {
                                        if candle.low <= trade.sl_price {
                                            if trade.is_be_applied
                                                && trade.sl_price == trade.entry_price
                                            {
                                                trade.outcome = Some("BE".to_string());
                                                trade.exit_price = Some(trade.sl_price);
                                                trade.pnl_r = Some(Decimal::ZERO);
                                            } else {
                                                trade.outcome = Some("LOSS".to_string());
                                                trade.exit_price = Some(trade.sl_price);
                                                trade.pnl_r = Some(Decimal::from(-1));
                                            }
                                            trade.exit_candle_idx = Some(candle_idx);
                                            trade.duration_candles = Some(current_duration);
                                            just_closed = true;
                                        } else if candle.high >= trade.tp_price {
                                            trade.outcome = Some("WIN".to_string());
                                            trade.exit_price = Some(trade.tp_price);
                                            let risk =
                                                (trade.entry_price - trade.original_sl_price).abs();
                                            let tp_r = if risk.is_zero() {
                                                Decimal::ZERO
                                            } else {
                                                (trade.tp_price - trade.entry_price) / risk
                                            };
                                            trade.pnl_r = Some(tp_r);
                                            trade.exit_candle_idx = Some(candle_idx);
                                            trade.duration_candles = Some(current_duration);
                                            just_closed = true;
                                        }
                                    }
                                    SignalType::SHORT => {
                                        if candle.high >= trade.sl_price {
                                            if trade.is_be_applied
                                                && trade.sl_price == trade.entry_price
                                            {
                                                trade.outcome = Some("BE".to_string());
                                                trade.exit_price = Some(trade.sl_price);
                                                trade.pnl_r = Some(Decimal::ZERO);
                                            } else {
                                                trade.outcome = Some("LOSS".to_string());
                                                trade.exit_price = Some(trade.sl_price);
                                                trade.pnl_r = Some(Decimal::from(-1));
                                            }
                                            trade.exit_candle_idx = Some(candle_idx);
                                            trade.duration_candles = Some(current_duration);
                                            just_closed = true;
                                        } else if candle.low <= trade.tp_price {
                                            trade.outcome = Some("WIN".to_string());
                                            trade.exit_price = Some(trade.tp_price);
                                            let risk =
                                                (trade.entry_price - trade.original_sl_price).abs();
                                            let tp_r = if risk.is_zero() {
                                                Decimal::ZERO
                                            } else {
                                                (trade.entry_price - trade.tp_price) / risk
                                            };
                                            trade.pnl_r = Some(tp_r);
                                            trade.exit_candle_idx = Some(candle_idx);
                                            trade.duration_candles = Some(current_duration);
                                            just_closed = true;
                                        }
                                    }
                                }
                            }

                            // T1.5: Record trade close - cooldown starts here
                            if just_closed {
                                engine.record_trade_close(symbol, interval, candle_idx);

                                // T11: Record trade result for kill switch (per symbol+TF, STICKY)
                                // PHASE A: Only count towards kill switch if bootstrap is complete
                                if ctx.bootstrap.is_complete() {
                                    let is_win = trade.outcome.as_deref() == Some("WIN");
                                    let ema50_slope = Some(ctx.get_ema50_slope());
                                    let current_atr = ctx.atr_14.current_value;
                                    engine.record_trade_result(
                                        symbol,
                                        interval,
                                        is_win,
                                        candle_idx,
                                        ema50_slope,
                                        current_atr,
                                    );
                                }

                                // Also record context-based close for multi-position
                                if let Some(ref ctx_id) = trade.context_id {
                                    engine.record_context_close(ctx_id, interval, candle_idx);
                                }

                                // Update position pool
                                for pool_trade in engine.get_position_pool_mut().active_trades_mut()
                                {
                                    if pool_trade.signal.signal_id == trade.signal.signal_id {
                                        let pnl = trade.pnl_r.unwrap_or(Decimal::ZERO);
                                        let exit = trade.exit_price.unwrap_or(candle.close);
                                        let outcome = trade.outcome.as_deref().unwrap_or("UNKNOWN");
                                        pool_trade.close(exit, pnl, outcome, candle_idx);
                                    }
                                }
                            }
                        }

                        candle_idx += 1;
                    }
                }
                Err(e) => error!("Data fetch failed: {}", e),
            }

            // Sonuçları Yazma
            if !trades.is_empty() {
                // Count all opened trades (including BE and MAX_DURATION)
                let opened_count = trades.len();
                let wins = trades
                    .iter()
                    .filter(|t| t.outcome.as_deref() == Some("WIN"))
                    .count();
                let losses = trades
                    .iter()
                    .filter(|t| t.outcome.as_deref() == Some("LOSS"))
                    .count();
                let total_pnl: Decimal = trades
                    .iter()
                    .filter(|t| t.outcome.is_some())
                    .map(|t| t.pnl_r.unwrap_or_default())
                    .sum();

                // Decisive trades = WIN + LOSS only (excludes BE and MAX_DURATION)
                let decisive_trades = wins + losses;
                let win_rate = if decisive_trades > 0 {
                    wins as f64 / decisive_trades as f64 * 100.0
                } else {
                    0.0
                };

                // T5.1: Build trade records for advanced metrics
                // Only include decisive trades (WIN/LOSS) for accurate metric calculation
                let trade_records: Vec<TradeRecord> = trades
                    .iter()
                    .filter(|t| {
                        t.outcome.as_deref() == Some("WIN") || t.outcome.as_deref() == Some("LOSS")
                    })
                    .map(|t| TradeRecord {
                        pnl_r: t.pnl_r.map(|d| d.to_f64().unwrap_or(0.0)).unwrap_or(0.0),
                        is_win: t.outcome.as_deref() == Some("WIN"),
                        duration_candles: t.duration_candles.unwrap_or(0),
                        regime: t.signal.regime_context.clone(),
                        confidence_tier: t.signal.confidence_tier.clone(),
                        // Multi-position fields
                        context_type: t.context_id.as_ref().map(|c| c.context_type.clone()),
                        opened_at_candle: Some(t.entry_candle_idx),
                        exit_candle_idx: t.exit_candle_idx,
                        adjusted_confidence: Some(t.adjusted_confidence),
                        was_concurrent: t.was_concurrent,
                    })
                    .collect();

                // T5.1: Calculate advanced metrics (on decisive trades only)
                let advanced_metrics = AdvancedMetrics::calculate(&trade_records);

                // T5.2: Generate regime report
                let regime_report = RegimeReport::generate(&trade_records);

                // Compute gross wins/losses R for correct overall Profit Factor
                let gross_wins_r: f64 = trades
                    .iter()
                    .filter(|t| t.outcome.as_deref() == Some("WIN"))
                    .map(|t| t.pnl_r.unwrap_or_default().to_f64().unwrap_or(0.0))
                    .sum();
                let gross_losses_r: f64 = trades
                    .iter()
                    .filter(|t| t.outcome.as_deref() == Some("LOSS"))
                    .map(|t| t.pnl_r.unwrap_or_default().to_f64().unwrap_or(0.0).abs())
                    .sum();

                // Now we can move trades into result
                let result = BacktestResult {
                    symbol: symbol.clone(),
                    timeframe: interval.clone(),
                    total_trades: opened_count,
                    wins,
                    losses,
                    win_rate,
                    total_pnl_r: total_pnl,
                    advanced_metrics: advanced_metrics.clone(),
                    regime_report,
                    signals: trades,
                };

                let filename = format!("{}/{}_{}_backtest.json", output_dir, symbol, interval);
                let mut file = File::create(&filename)?;
                let json = serde_json::to_string_pretty(&result)?;
                file.write_all(json.as_bytes())?;

                // T5.1: Enhanced logging with advanced metrics
                info!(
                    "📊 Rapor: {} {} -> PnL: {}R (%{:.1} WR, {}/{} W/L, {} opened)",
                    symbol, interval, total_pnl, win_rate, wins, losses, opened_count
                );
                info!(
                    "   📈 Expectancy: {:.3}R | PF: {:.2} | Sharpe: {:.2}",
                    advanced_metrics.expectancy_r,
                    advanced_metrics.profit_factor,
                    advanced_metrics.sharpe_ratio_approx
                );
                info!(
                    "   📉 Max Consec Loss: {} | Avg Duration: {:.1} candles",
                    advanced_metrics.max_consecutive_losses,
                    advanced_metrics.avg_trade_duration_candles
                );

                // Track for summary
                let pair_key = format!("{} {}", symbol, interval);
                let pair_result = PairResult {
                    symbol: symbol.clone(),
                    timeframe: interval.clone(),
                    trades: decisive_trades,     // WIN + LOSS only
                    opened_trades: opened_count, // all opened
                    wins,
                    losses,
                    win_rate,
                    pnl_r: total_pnl,
                    expectancy: advanced_metrics.expectancy_r,
                    profit_factor: advanced_metrics.profit_factor,
                    sharpe: advanced_metrics.sharpe_ratio_approx,
                    max_consec_loss: advanced_metrics.max_consecutive_losses,
                    gross_wins_r,
                    gross_losses_r,
                };

                // Track best/worst
                if total_pnl > best_pnl {
                    best_pnl = total_pnl;
                    summary.best_performer = Some(pair_key.clone());
                }
                if total_pnl < worst_pnl {
                    worst_pnl = total_pnl;
                    summary.worst_performer = Some(pair_key.clone());
                }

                // Accumulate totals
                // total_trades = decisive (WIN+LOSS), total_opened_trades = all opened
                summary.total_trades += decisive_trades;
                summary.total_opened_trades += opened_count;
                summary.total_wins += wins;
                summary.total_losses += losses;
                summary.overall_pnl_r += total_pnl;

                all_results.push(pair_result);
            } else {
                info!("ℹ️  Sinyal bulunamadı: {} {}", symbol, interval);
            }

            summary.total_timeframes_tested += 1;
        }
    }

    // Merge engine block stats
    summary.block_stats = engine.get_stats().clone();
    summary.total_symbols = symbols.len();
    summary.results_by_pair = all_results;

    // Calculate overall metrics
    if summary.total_trades > 0 {
        // Win rate: wins / (wins + losses) — excludes BE and MAX_DURATION
        let decisive_total = summary.total_wins + summary.total_losses;
        summary.overall_win_rate = if decisive_total > 0 {
            summary.total_wins as f64 / decisive_total as f64 * 100.0
        } else {
            0.0
        };

        // Expectancy: total PnL / decisive trades (WIN+LOSS only)
        // Using decisive_total so BE/MAX_DURATION don't dilute the expectancy
        summary.overall_expectancy = if decisive_total > 0 {
            summary.overall_pnl_r.to_f64().unwrap_or(0.0) / decisive_total as f64
        } else {
            0.0
        };

        // Profit Factor: sum of all winning R / sum of all losing R (trade-by-trade, not pair-level)
        // We accumulate gross_wins_r and gross_losses_r per pair for this purpose
        let gross_wins: f64 = summary.results_by_pair.iter().map(|r| r.gross_wins_r).sum();
        let gross_losses: f64 = summary
            .results_by_pair
            .iter()
            .map(|r| r.gross_losses_r)
            .sum();
        summary.overall_profit_factor = if gross_losses > 0.0 {
            gross_wins / gross_losses
        } else if gross_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }

    // Calculate dollar P&L with different risk percentages
    let starting_capital_per_pair = 1000.0;
    let total_starting_capital = starting_capital_per_pair * summary.results_by_pair.len() as f64;
    let risk_percentages = vec![1.0, 2.0, 3.0];
    let mut risk_scenarios = Vec::new();

    for risk_pct in risk_percentages {
        let risk_per_r = starting_capital_per_pair * (risk_pct / 100.0);
        let mut pair_results = Vec::new();
        let mut total_dollar_pnl = 0.0;

        for result in &summary.results_by_pair {
            let pnl_r = result.pnl_r.to_f64().unwrap_or(0.0);
            let dollar_pnl = pnl_r * risk_per_r;
            let final_capital = starting_capital_per_pair + dollar_pnl;
            let return_percent = (dollar_pnl / starting_capital_per_pair) * 100.0;

            pair_results.push(DollarPairResult {
                pair: format!("{} {}", result.symbol, result.timeframe),
                starting_capital: starting_capital_per_pair,
                dollar_pnl,
                final_capital,
                return_percent,
            });

            total_dollar_pnl += dollar_pnl;
        }

        let total_final_capital = total_starting_capital + total_dollar_pnl;
        let total_return_percent = (total_dollar_pnl / total_starting_capital) * 100.0;

        risk_scenarios.push(RiskScenario {
            risk_percent: risk_pct,
            risk_per_r,
            total_dollar_pnl,
            total_final_capital,
            total_return_percent,
            pair_results,
        });
    }

    summary.dollar_pnl = Some(DollarPnlSummary {
        starting_capital_per_pair,
        total_starting_capital,
        risk_scenarios,
    });

    // Print final summary
    print_summary(&summary);

    // Save summary to file
    let summary_filename = format!("{}/BACKTEST_SUMMARY.json", output_dir);
    let mut summary_file = File::create(&summary_filename)?;
    let summary_json = serde_json::to_string_pretty(&summary)?;
    summary_file.write_all(summary_json.as_bytes())?;

    info!("💾 Summary saved to: {}", summary_filename);
    Ok(())
}

fn print_summary(summary: &BacktestSummary) {
    info!("");
    info!("═══════════════════════════════════════════════════════════════");
    info!("                    🏁 BACKTEST SUMMARY REPORT                  ");
    info!("═══════════════════════════════════════════════════════════════");
    info!("");

    // Overall Performance
    info!("📊 OVERALL PERFORMANCE");
    info!("───────────────────────────────────────────────────────────────");
    info!(
        "   Opened Trades:    {} (all signals taken)",
        summary.total_opened_trades
    );
    info!(
        "   Decisive Trades:  {} (WIN + LOSS only, excl. BE/MAX_DUR)",
        summary.total_trades
    );
    info!(
        "   Wins: {} | Losses: {}",
        summary.total_wins, summary.total_losses
    );
    info!(
        "   Win Rate: {:.1}%  (of decisive trades)",
        summary.overall_win_rate
    );
    info!("   Total PnL: {}R", summary.overall_pnl_r);
    info!(
        "   Expectancy: {:.3}R per decisive trade",
        summary.overall_expectancy
    );
    info!(
        "   Profit Factor: {:.2}  (gross wins R / gross losses R)",
        summary.overall_profit_factor
    );
    info!("");

    // Best/Worst performers
    if let Some(ref best) = summary.best_performer {
        info!("   🏆 Best Performer: {}", best);
    }
    if let Some(ref worst) = summary.worst_performer {
        info!("   ⚠️  Worst Performer: {}", worst);
    }
    info!("");

    // Block Statistics
    info!("🛡️  BLOCK STATISTICS (Filter Performance)");
    info!("───────────────────────────────────────────────────────────────");
    let bs = &summary.block_stats;
    info!("   Total Evaluations: {}", bs.total_evaluations);
    info!(
        "   Signals Generated: {} ({:.2}% signal rate)",
        bs.total_signals_generated,
        bs.signal_rate()
    );
    info!("   Total Blocks: {}", bs.total_blocks());
    info!("");
    info!("   📍 Block Breakdown:");
    info!("      Wick Trap:        {:>6} blocks", bs.wick_trap_blocks);
    info!("      Flat EMA:         {:>6} blocks", bs.flat_ema_blocks);
    info!("      Low ATR:          {:>6} blocks", bs.low_atr_blocks);
    info!(
        "      Bootstrap:        {:>6} blocks",
        bs.bootstrap_incomplete
    );
    info!(
        "      Open Trade:       {:>6} blocks (legacy single-position)",
        bs.open_trade_blocks
    );
    info!(
        "      Cooldown:         {:>6} blocks (post-close cooldown)",
        bs.cooldown_blocks
    );
    info!("      Score Too Low:    {:>6} blocks", bs.score_too_low);
    info!("      Policy Blocked:   {:>6} blocks", bs.policy_blocked);
    info!("      LSTM Filtered:    {:>6} blocks", bs.lstm_filtered);
    info!("");
    info!("   📍 Multi-Position Blocks (Phase 7):");
    info!(
        "      Max Trades:       {:>6} blocks (reached max concurrent)",
        bs.max_trades_reached
    );
    info!(
        "      Duplicate Ctx:    {:>6} blocks (same context_id)",
        bs.duplicate_context
    );
    info!(
        "      Hedge Blocked:    {:>6} blocks (opposite direction)",
        bs.hedge_blocked
    );
    info!(
        "      Context CD:       {:>6} blocks (context-specific cooldown)",
        bs.context_cooldown_blocks
    );
    info!("");
    info!("   📍 Phase 8 Blocks:");
    info!(
        "      Trend Saturation: {:>6} blocks (slope weakening)",
        bs.trend_saturation_blocks
    );
    info!(
        "      Weak Replaced:    {:>6} trades (replaced by stronger)",
        bs.weak_trade_replaced
    );
    info!("");
    info!("   📍 Phase 9 Exit Stats:");
    info!(
        "      Max Duration:     {:>6} exits (forced after max duration)",
        bs.max_duration_exits
    );
    info!(
        "      BE Applied:       {:>6} trades (moved SL to entry)",
        bs.be_applied_count
    );
    info!(
        "      Partial TP:       {:>6} times (50% closed at 1R)",
        bs.partial_tp_count
    );
    info!("");
    info!("   📍 Phase 10 Safety:");
    info!(
        "      Kill Switch:      {:>6} triggers (consec losses)",
        bs.kill_switch_triggered
    );
    info!("");

    // Results by pair table
    info!("📋 RESULTS BY PAIR");
    info!("───────────────────────────────────────────────────────────────");
    info!(
        "   {:12} {:>6} {:>6} {:>6} {:>7} {:>8} {:>8} {:>7}",
        "PAIR", "OPENED", "W+L", "WR%", "PnL(R)", "EXPECT", "PF", "SHARPE"
    );
    info!(
        "   {:─<12} {:─>6} {:─>6} {:─>6} {:─>7} {:─>8} {:─>8} {:─>7}",
        "", "", "", "", "", "", "", ""
    );

    for r in &summary.results_by_pair {
        let pair_name = format!("{} {}", r.symbol.replace("USDT", ""), r.timeframe);
        info!(
            "   {:12} {:>6} {:>6} {:>5.1}% {:>7} {:>7.3}R {:>7.2} {:>7.2}",
            pair_name,
            r.opened_trades,
            r.trades,
            r.win_rate,
            r.pnl_r,
            r.expectancy,
            r.profit_factor,
            r.sharpe
        );
    }

    info!("");

    // Dollar P&L Summary
    if let Some(ref dollar_pnl) = summary.dollar_pnl {
        info!("💰 DOLLAR P&L CALCULATION");
        info!("───────────────────────────────────────────────────────────────");
        info!(
            "   Starting Capital per Pair: ${:.2}",
            dollar_pnl.starting_capital_per_pair
        );
        info!(
            "   Total Starting Capital: ${:.2}",
            dollar_pnl.total_starting_capital
        );
        info!("");

        for scenario in &dollar_pnl.risk_scenarios {
            info!(
                "   📊 Risk {}% per trade (${:.2} per R):",
                scenario.risk_percent, scenario.risk_per_r
            );
            info!("      Total P&L: ${:.2}", scenario.total_dollar_pnl);
            info!("      Final Capital: ${:.2}", scenario.total_final_capital);
            info!("      Total Return: {:.2}%", scenario.total_return_percent);
            info!("");
        }
    }

    info!("═══════════════════════════════════════════════════════════════");
    info!("                         🏁 END REPORT                         ");
    info!("═══════════════════════════════════════════════════════════════");
    info!("");
}

fn calculate_sl_tp(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    entry: Decimal,
) -> (Decimal, Decimal) {
    let atr = ctx.atr_14.current_value.unwrap_or(Decimal::ONE);

    // ORDER BLOCK TABANLI TP/SL
    // OB tracker varsa ve geçerli OB'ler mevcutsa, OB tabanlı hesapla
    // Fallback olarak pivot seviyeleri kullanılır
    ctx.ob_tracker.calculate_ob_sl_tp(
        &signal.signal,
        entry,
        atr,
        ctx.structure.last_pivot_low,
        ctx.structure.last_pivot_high,
        &ctx.pivot_high_history,
        &ctx.pivot_low_history,
    )
}

// =============================================================================
// LOCAL CSV BACKTEST
// =============================================================================

/// Parse a CSV file and return candles
fn parse_csv_candles(file_path: &str) -> anyhow::Result<Vec<Candle>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut candles = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip header
        if idx == 0 && line.starts_with("timestamp") {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        // Parse timestamp: "2017-08-17 04:00:00+00:00"
        let ts_str = parts[0].trim();
        let timestamp = parse_timestamp(ts_str)?;

        let open = Decimal::from_str(parts[1].trim())?;
        let high = Decimal::from_str(parts[2].trim())?;
        let low = Decimal::from_str(parts[3].trim())?;
        let close = Decimal::from_str(parts[4].trim())?;
        let volume = Decimal::from_str(parts[5].trim())?;

        candles.push(Candle {
            open_time: timestamp,
            close_time: Some(timestamp + Duration::hours(1)),
            open,
            high,
            low,
            close,
            volume,
        });
    }

    info!("📂 Loaded {} candles from CSV", candles.len());
    Ok(candles)
}

fn parse_timestamp(ts: &str) -> anyhow::Result<DateTime<Utc>> {
    // Handle format: "2017-08-17 04:00:00+00:00"
    let clean = ts.replace("+00:00", "").replace("UTC", "");
    let naive = NaiveDateTime::parse_from_str(clean.trim(), "%Y-%m-%d %H:%M:%S")?;
    Ok(Utc.from_utc_datetime(&naive))
}

/// Run backtest on local CSV file
pub async fn run_csv_backtest(
    csv_path: &str,
    symbol: &str,
    timeframe: &str,
    exit_mode: &str,
    send_alpaca_signals: bool,
    output_dir: &str,
    lstm_filter: Option<LstmFilter>,
    lstm_mode: LstmMode,
    pool_overrides: Option<PoolConfigOverrides>,
) -> anyhow::Result<()> {
    info!("═══════════════════════════════════════════════════════════════");
    info!("   🗂️  LOCAL CSV BACKTEST: {} ({})", symbol, timeframe);
    info!("   📁 File: {}", csv_path);
    info!("═══════════════════════════════════════════════════════════════");

    // Klasörü oluştur
    fs::create_dir_all(output_dir)?;

    // Load candles from CSV
    let candles = parse_csv_candles(csv_path)?;

    if candles.is_empty() {
        error!("❌ No candles loaded from CSV");
        return Ok(());
    }

    info!(
        "📅 Date Range: {} to {}",
        candles.first().unwrap().open_time.format("%Y-%m-%d"),
        candles.last().unwrap().open_time.format("%Y-%m-%d")
    );

    let mut engine = SignalEngine::new_backtest_mode(); // T1.3: Backtest mode with shorter cooldowns
    if let Some(filter) = lstm_filter {
        engine.set_lstm_filter(filter);
    }
    engine.set_lstm_mode(lstm_mode);
    apply_pool_config_overrides(&mut engine, pool_overrides);
    let exit_mode = ExitMode::from_str(exit_mode);
    let _ = send_alpaca_signals;
    let mut ctx = SymbolContext::new(symbol.to_string(), timeframe.to_string());
    let mut trades: Vec<SimulatedTrade> = Vec::new();
    let mut candle_idx: usize = 0;

    // Progress tracking
    let total_candles = candles.len();
    let report_interval = total_candles / 10;

    for candle in candles.iter() {
        // Progress report
        if candle_idx > 0 && candle_idx % report_interval == 0 {
            let progress = (candle_idx as f64 / total_candles as f64) * 100.0;
            info!(
                "   ⏳ Progress: {:.0}% ({}/{} candles)",
                progress, candle_idx, total_candles
            );
        }

        // 1. Add candle
        ctx.add_candle(candle.clone());

        // T11.2: Try to reset kill switch if conditions are met
        let current_ema50_slope = Some(ctx.get_ema50_slope());
        let current_atr = ctx.atr_14.current_value;
        let median_atr = Some(ctx.get_median_atr_ratio() * current_atr.unwrap_or(Decimal::ONE)); // Approximate
        engine.try_reset_kill_switch(
            symbol,
            timeframe,
            candle_idx,
            current_ema50_slope,
            current_atr,
            median_atr,
        );

        // 2. Check for signal
        if let Some(signal) = engine.evaluate(&mut ctx) {
            let entry = candle.close;
            let (sl, tp) = calculate_sl_tp(&signal, &ctx, entry);

            // Get context ID from context (set during evaluate)
            let context_id = ctx.current_context_id.clone();
            let adjusted_confidence = signal.confidence;

            // T8.2: Get context score from signal
            let context_score = signal.confidence as i32;

            // T8.3: Get EMA50 slope at entry
            let ema50_slope = ctx.get_ema50_slope();

            // Check if this trade overlaps with existing trades
            let active_count = engine.get_position_pool().active_count(symbol, timeframe);
            let was_concurrent = active_count > 0;

            // T1.4: Record trade open (signal generated = trade entered)
            engine.record_trade_open(symbol, timeframe, candle_idx);

            // Create ActiveTrade for position pool
            if let Some(ref ctx_id) = context_id {
                let active_trade =
                    ActiveTrade::new(signal.clone(), entry, sl, tp, ctx_id.clone(), candle_idx)
                        .with_context_score(context_score)
                        .with_ema50_slope(ema50_slope);
                engine.add_trade_to_pool(active_trade);
            }

            trades.push(SimulatedTrade {
                signal,
                entry_price: entry,
                sl_price: sl,
                original_sl_price: sl,
                tp_price: tp,
                exit_price: None,
                pnl_r: None,
                outcome: None,
                entry_candle_idx: candle_idx,
                exit_candle_idx: None,
                duration_candles: None,
                context_id,
                adjusted_confidence,
                was_concurrent,
                context_score,
                ema50_slope_at_entry: Some(ema50_slope),
                is_be_applied: false,
            });
        }

        // 3. Manage open positions
        let pool_config = engine.get_position_pool().config.clone();
        let allow_supertrend_exit = exit_mode != ExitMode::SlTp;
        let allow_sl_tp_exit = exit_mode != ExitMode::Supertrend;

        for trade in trades.iter_mut() {
            if trade.outcome.is_some() {
                continue;
            }

            if trade.signal.timestamp == candle.close_time.unwrap_or(candle.open_time) {
                continue;
            }

            let mut just_closed = false;
            let current_duration = (candle_idx - trade.entry_candle_idx) as u32;

            // T9.1: Max Duration HARD CAP - Force exit after max_trade_duration_candles
            if current_duration >= pool_config.max_trade_duration_candles {
                // Calculate PnL at current price
                let risk = (trade.entry_price - trade.original_sl_price).abs();
                let unrealized_pnl = match trade.signal.signal {
                    SignalType::LONG => (candle.close - trade.entry_price) / risk,
                    SignalType::SHORT => (trade.entry_price - candle.close) / risk,
                };

                trade.outcome = Some("MAX_DURATION".to_string());
                trade.exit_price = Some(candle.close);
                trade.pnl_r = Some(unrealized_pnl);
                trade.exit_candle_idx = Some(candle_idx);
                trade.duration_candles = Some(current_duration);
                just_closed = true;
                engine.block_stats.max_duration_exits += 1;
            }

            if !just_closed && allow_supertrend_exit {
                let reversal = match trade.signal.signal {
                    SignalType::LONG => ctx.pine_trend_changed_bearish,
                    SignalType::SHORT => ctx.pine_trend_changed_bullish,
                };

                if reversal {
                    let risk = (trade.entry_price - trade.original_sl_price).abs();
                    let pnl_r = if risk.is_zero() {
                        Decimal::ZERO
                    } else {
                        match trade.signal.signal {
                            SignalType::LONG => (candle.close - trade.entry_price) / risk,
                            SignalType::SHORT => (trade.entry_price - candle.close) / risk,
                        }
                    };

                    trade.outcome = Some(
                        if pnl_r > Decimal::ZERO {
                            "WIN"
                        } else if pnl_r < Decimal::ZERO {
                            "LOSS"
                        } else {
                            "BE"
                        }
                        .to_string(),
                    );
                    trade.exit_price = Some(candle.close);
                    trade.pnl_r = Some(pnl_r);
                    trade.exit_candle_idx = Some(candle_idx);
                    trade.duration_candles = Some(current_duration);
                    just_closed = true;
                }
            }

            // T9.2: Time-based BE - Move SL to entry after be_threshold_candles if profit < be_min_profit_r
            if allow_sl_tp_exit
                && !just_closed
                && !trade.is_be_applied
                && current_duration >= pool_config.be_threshold_candles
            {
                let risk = (trade.entry_price - trade.original_sl_price).abs();
                let unrealized_r = match trade.signal.signal {
                    SignalType::LONG => (candle.close - trade.entry_price) / risk,
                    SignalType::SHORT => (trade.entry_price - candle.close) / risk,
                };

                // Only apply BE if trade is NOT doing well (< 0.5R profit)
                if unrealized_r < pool_config.be_min_profit_r {
                    trade.sl_price = trade.entry_price; // Move SL to entry (break-even)
                    trade.is_be_applied = true;
                    engine.block_stats.be_applied_count += 1;
                }
            }

            // Normal SL/TP checks (if not already closed)
            if allow_sl_tp_exit && !just_closed {
                match trade.signal.signal {
                    SignalType::LONG => {
                        if candle.low <= trade.sl_price {
                            // Check if it's a BE exit or regular loss
                            if trade.is_be_applied && trade.sl_price == trade.entry_price {
                                trade.outcome = Some("BE".to_string());
                                trade.exit_price = Some(trade.sl_price);
                                trade.pnl_r = Some(Decimal::ZERO); // Break-even = 0R
                            } else {
                                trade.outcome = Some("LOSS".to_string());
                                trade.exit_price = Some(trade.sl_price);
                                trade.pnl_r = Some(Decimal::from(-1));
                            }
                            trade.exit_candle_idx = Some(candle_idx);
                            trade.duration_candles = Some(current_duration);
                            just_closed = true;
                        } else if candle.high >= trade.tp_price {
                            trade.outcome = Some("WIN".to_string());
                            trade.exit_price = Some(trade.tp_price);
                            trade.pnl_r = Some(Decimal::from_f64(1.5).unwrap());
                            trade.exit_candle_idx = Some(candle_idx);
                            trade.duration_candles = Some(current_duration);
                            just_closed = true;
                        }
                    }
                    SignalType::SHORT => {
                        if candle.high >= trade.sl_price {
                            if trade.is_be_applied && trade.sl_price == trade.entry_price {
                                trade.outcome = Some("BE".to_string());
                                trade.exit_price = Some(trade.sl_price);
                                trade.pnl_r = Some(Decimal::ZERO);
                            } else {
                                trade.outcome = Some("LOSS".to_string());
                                trade.exit_price = Some(trade.sl_price);
                                trade.pnl_r = Some(Decimal::from(-1));
                            }
                            trade.exit_candle_idx = Some(candle_idx);
                            trade.duration_candles = Some(current_duration);
                            just_closed = true;
                        } else if candle.low <= trade.tp_price {
                            trade.outcome = Some("WIN".to_string());
                            trade.exit_price = Some(trade.tp_price);
                            trade.pnl_r = Some(Decimal::from_f64(1.5).unwrap());
                            trade.exit_candle_idx = Some(candle_idx);
                            trade.duration_candles = Some(current_duration);
                            just_closed = true;
                        }
                    }
                }
            }

            // T1.5: Record trade close - THIS IS WHERE COOLDOWN STARTS
            if just_closed {
                engine.record_trade_close(symbol, timeframe, candle_idx);

                // T11: Record trade result for kill switch (per symbol+TF, STICKY)
                // PHASE A: Only count towards kill switch if bootstrap is complete
                if ctx.bootstrap.is_complete() {
                    let is_win = trade.outcome.as_deref() == Some("WIN");
                    let ema50_slope = Some(ctx.get_ema50_slope());
                    let current_atr = ctx.atr_14.current_value;
                    engine.record_trade_result(
                        symbol,
                        timeframe,
                        is_win,
                        candle_idx,
                        ema50_slope,
                        current_atr,
                    );
                }

                // Also record context-based close for multi-position
                if let Some(ref ctx_id) = trade.context_id {
                    engine.record_context_close(ctx_id, timeframe, candle_idx);
                }

                // Update position pool
                for pool_trade in engine.get_position_pool_mut().active_trades_mut() {
                    if pool_trade.signal.signal_id == trade.signal.signal_id {
                        let pnl = trade.pnl_r.unwrap_or(Decimal::ZERO);
                        let exit = trade.exit_price.unwrap_or(candle.close);
                        let outcome = trade.outcome.as_deref().unwrap_or("UNKNOWN");
                        pool_trade.close(exit, pnl, outcome, candle_idx);
                    }
                }
            }
        }

        candle_idx += 1;
    }

    // Calculate results
    let completed_trades: Vec<&SimulatedTrade> =
        trades.iter().filter(|t| t.outcome.is_some()).collect();

    let completed_count = completed_trades.len();
    let wins = completed_trades
        .iter()
        .filter(|t| t.outcome.as_deref() == Some("WIN"))
        .count();
    let losses = completed_trades
        .iter()
        .filter(|t| t.outcome.as_deref() == Some("LOSS"))
        .count();
    let be_count = completed_trades
        .iter()
        .filter(|t| t.outcome.as_deref() == Some("BE"))
        .count();
    let max_dur_count = completed_trades
        .iter()
        .filter(|t| t.outcome.as_deref() == Some("MAX_DURATION"))
        .count();
    // Win rate should only count WIN/LOSS, not BE or MAX_DURATION
    let decisive_trades = wins + losses;
    let win_rate = if decisive_trades > 0 {
        (wins as f64 / decisive_trades as f64) * 100.0
    } else {
        0.0
    };

    let total_pnl: Decimal = completed_trades.iter().filter_map(|t| t.pnl_r).sum();

    let expectancy = if completed_count > 0 {
        total_pnl.to_f64().unwrap_or(0.0) / completed_count as f64
    } else {
        0.0
    };

    // T12.2: FIXED Profit Factor = sum(positive_R) / abs(sum(negative_R))
    // NOT (wins * 1.5) / losses - that's WRONG!
    let positive_r_sum: Decimal = completed_trades
        .iter()
        .filter_map(|t| t.pnl_r)
        .filter(|r| *r > Decimal::ZERO)
        .sum();
    let negative_r_sum: Decimal = completed_trades
        .iter()
        .filter_map(|t| t.pnl_r)
        .filter(|r| *r < Decimal::ZERO)
        .sum::<Decimal>()
        .abs();

    let profit_factor = if negative_r_sum > Decimal::ZERO {
        (positive_r_sum / negative_r_sum).to_f64().unwrap_or(0.0)
    } else if positive_r_sum > Decimal::ZERO {
        f64::INFINITY
    } else {
        0.0
    };

    // Calculate Sharpe-like ratio
    let returns: Vec<f64> = completed_trades
        .iter()
        .filter_map(|t| t.pnl_r.map(|r| r.to_f64().unwrap_or(0.0)))
        .collect();

    let sharpe = if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean / std_dev
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Max consecutive losses
    let mut max_consec = 0u32;
    let mut current_consec = 0u32;
    for trade in &completed_trades {
        if trade.outcome.as_ref().unwrap() == "LOSS" {
            current_consec += 1;
            max_consec = max_consec.max(current_consec);
        } else {
            current_consec = 0;
        }
    }

    // Avg duration
    let durations: Vec<u32> = completed_trades
        .iter()
        .filter_map(|t| t.duration_candles)
        .collect();
    let avg_duration = if !durations.is_empty() {
        durations.iter().sum::<u32>() as f64 / durations.len() as f64
    } else {
        0.0
    };

    // T12.3: Sanity Check Guard - warn if metrics are inconsistent
    if profit_factor < 1.0 && total_pnl > Decimal::ZERO {
        warn!(
            "⚠️ METRIC INCONSISTENCY: PF={:.2} < 1 but PnL={}R > 0. Check R accounting!",
            profit_factor, total_pnl
        );
    }
    if profit_factor > 1.0 && total_pnl < Decimal::ZERO {
        warn!(
            "⚠️ METRIC INCONSISTENCY: PF={:.2} > 1 but PnL={}R < 0. Check R accounting!",
            profit_factor, total_pnl
        );
    }

    // Block stats
    let block_stats = engine.get_stats();

    // Print Summary
    info!("");
    info!("═══════════════════════════════════════════════════════════════");
    info!("              🏁 ALL-TIME BACKTEST SUMMARY: {}", symbol);
    info!("═══════════════════════════════════════════════════════════════");
    info!("");
    info!("📈 PERFORMANCE");
    info!("───────────────────────────────────────────────────────────────");
    info!("   Total Candles Processed: {}", total_candles);
    info!("   Total Signals Generated: {}", trades.len());
    info!(
        "   Completed Trades: {} (Wins: {} | Losses: {} | BE: {} | MaxDur: {})",
        completed_count, wins, losses, be_count, max_dur_count
    );
    info!(
        "   Win Rate: {:.1}% (of decisive trades: {})",
        win_rate, decisive_trades
    );
    info!("   Total PnL: {}R", total_pnl);
    info!("   Expectancy: {:.3}R per trade", expectancy);
    info!("   Profit Factor: {:.2}", profit_factor);
    info!("   Sharpe Ratio: {:.2}", sharpe);
    info!("   Max Consecutive Losses: {}", max_consec);
    info!("   Avg Trade Duration: {:.1} candles", avg_duration);
    info!("");
    info!("� R ACCOUNTING VERIFICATION");
    info!("───────────────────────────────────────────────────────────────");
    info!("   Gross Profit (sum +R): {:.2}R", positive_r_sum);
    info!("   Gross Loss (sum -R):   {:.2}R", negative_r_sum);
    info!("   Net PnL (diff):        {:.2}R", total_pnl);
    info!("   PF (gross/abs(loss)):  {:.2}", profit_factor);
    info!("");
    info!("�🛡️ BLOCK STATISTICS");
    info!("───────────────────────────────────────────────────────────────");
    info!("   Total Evaluations: {}", block_stats.total_evaluations);
    info!(
        "   Signals Generated: {} ({:.2}% signal rate)",
        block_stats.total_signals_generated,
        block_stats.signal_rate()
    );
    info!("   Total Blocks: {}", block_stats.total_blocks());
    info!("");
    info!("   📊 Block Breakdown:");
    info!(
        "      Wick Trap:        {:>6} blocks",
        block_stats.wick_trap_blocks
    );
    info!(
        "      Flat EMA:         {:>6} blocks",
        block_stats.flat_ema_blocks
    );
    info!(
        "      Low ATR:          {:>6} blocks",
        block_stats.low_atr_blocks
    );
    info!(
        "      Bootstrap:        {:>6} blocks",
        block_stats.bootstrap_incomplete
    );
    info!(
        "      Open Trade:       {:>6} blocks (legacy single-position)",
        block_stats.open_trade_blocks
    );
    info!(
        "      Cooldown:         {:>6} blocks (post-close cooldown)",
        block_stats.cooldown_blocks
    );
    info!(
        "      Score Too Low:    {:>6} blocks",
        block_stats.score_too_low
    );
    info!(
        "      Policy Blocked:   {:>6} blocks",
        block_stats.policy_blocked
    );
    info!(
        "      LSTM Filtered:    {:>6} blocks",
        block_stats.lstm_filtered
    );
    info!("");
    info!("   📊 Multi-Position Blocks:");
    info!(
        "      Max Trades:       {:>6} blocks",
        block_stats.max_trades_reached
    );
    info!(
        "      Duplicate Ctx:    {:>6} blocks",
        block_stats.duplicate_context
    );
    info!(
        "      Hedge Blocked:    {:>6} blocks",
        block_stats.hedge_blocked
    );
    info!(
        "      Context CD:       {:>6} blocks",
        block_stats.context_cooldown_blocks
    );
    info!("");

    // Multi-position metrics
    let pool = engine.get_position_pool();
    info!("   📊 Multi-Position Metrics:");
    info!(
        "      Max Concurrent:   {:>6} trades",
        pool.max_concurrent_trades()
    );
    info!(
        "      Avg Concurrent:   {:>6.2} trades",
        pool.avg_concurrent_trades()
    );
    info!("");
    info!("═══════════════════════════════════════════════════════════════");

    // Save results to JSON
    let result = BacktestResult {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        total_trades: completed_count,
        wins,
        losses,
        win_rate,
        total_pnl_r: total_pnl,
        advanced_metrics: AdvancedMetrics {
            expectancy_r: expectancy,
            profit_factor,
            sharpe_ratio_approx: sharpe,
            max_consecutive_losses: max_consec,
            avg_trade_duration_candles: avg_duration,
            trade_count: completed_count as u32,
            ..Default::default()
        },
        regime_report: RegimeReport::default(),
        signals: trades,
    };

    let file_name = format!("{}/{}_all_time_backtest.json", output_dir, symbol);
    let json = serde_json::to_string_pretty(&result)?;
    let mut file = File::create(&file_name)?;
    file.write_all(json.as_bytes())?;
    info!("📁 Results saved to: {}", file_name);

    Ok(())
}
