use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Duration, Utc, TimeZone, NaiveDateTime};
use std::fs::{self, File};
use std::io::{Write, BufRead, BufReader};
use std::path::Path;
use crate::types::{TradeSignal, Candle, SignalType, TrendState, RegimeContext, ContextId, ActiveTrade, PositionPool, PositionPoolConfig};
use crate::state::SymbolContext;
use crate::engine::SignalEngine;
use crate::connect::BinanceClient;
use crate::policy::TimeframePolicy;
use crate::analytics::{
    AdvancedMetrics, TradeRecord, RegimeReport, ExtendedBacktestResult, BlockStats
};
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

// =============================================================================
// BACKTEST SUMMARY REPORT
// =============================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct BacktestSummary {
    pub total_symbols: usize,
    pub total_timeframes_tested: usize,
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
}

#[derive(Debug, Clone, Serialize)]
pub struct PairResult {
    pub symbol: String,
    pub timeframe: String,
    pub trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub pnl_r: Decimal,
    pub expectancy: f64,
    pub profit_factor: f64,
    pub sharpe: f64,
    pub max_consec_loss: u32,
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
    output_dir: &str
) -> anyhow::Result<()> {
    info!("🔄 Backtest Başlatılıyor... (Son {} gün)", days);
    info!("═══════════════════════════════════════════════════════════════");
    
    // Klasörü oluştur
    fs::create_dir_all(output_dir)?;
    
    let client = BinanceClient::new();
    let mut engine = SignalEngine::new_backtest_mode(); // Backtest mode: bypasses policy
    let policy = TimeframePolicy::new();
    
    // Summary tracking
    let mut summary = BacktestSummary::default();
    let mut all_results: Vec<PairResult> = Vec::new();
    let mut best_pnl = Decimal::MIN;
    let mut worst_pnl = Decimal::MAX;
    
    // Binance limit: 1000 candles per call. 
    // Basitlik için backtest şimdilik son 1000 mum ile sınırlı, 
    // ama pagination ile geriye gidilebilir.
    let limit = 1000; 

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
            match client.fetch_candles(symbol, interval, limit).await {
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
                            let active_count = engine.get_position_pool().active_count(symbol, interval);
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
                        
                        for trade in trades.iter_mut() {
                            if trade.outcome.is_some() { continue; } // Zaten kapandı

                            // Trade sinyal mumuyla aynıysa skip et
                            if trade.signal.timestamp == candle.close_time.unwrap_or(candle.open_time) {
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
                            
                            // T9.2: Time-based BE
                            if !just_closed && !trade.is_be_applied && current_duration >= pool_config.be_threshold_candles {
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
                            if !just_closed {
                                match trade.signal.signal {
                                    SignalType::LONG => {
                                        if candle.low <= trade.sl_price {
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
                                        } else if candle.high >= trade.tp_price {
                                            trade.outcome = Some("WIN".to_string());
                                            trade.exit_price = Some(trade.tp_price);
                                            trade.pnl_r = Some(Decimal::from_f64(1.5).unwrap());
                                            trade.exit_candle_idx = Some(candle_idx);
                                            trade.duration_candles = Some(current_duration);
                                            just_closed = true;
                                        }
                                    },
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
                            
                            // T1.5: Record trade close - cooldown starts here
                            if just_closed {
                                engine.record_trade_close(symbol, interval, candle_idx);
                                
                                // T11: Record trade result for kill switch (per symbol+TF, STICKY)
                                let is_win = trade.outcome.as_deref() == Some("WIN");
                                let ema50_slope = Some(ctx.get_ema50_slope());
                                let current_atr = ctx.atr_14.current_value;
                                engine.record_trade_result(symbol, interval, is_win, candle_idx, ema50_slope, current_atr);
                                
                                // Also record context-based close for multi-position
                                if let Some(ref ctx_id) = trade.context_id {
                                    engine.record_context_close(ctx_id, interval, candle_idx);
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
                },
                Err(e) => error!("Data fetch failed: {}", e),
            }
            
            // Sonuçları Yazma
            if !trades.is_empty() {
                // First calculate stats from references
                let completed_count = trades.iter().filter(|t| t.outcome.is_some()).count();
                let wins = trades.iter().filter(|t| t.outcome.as_deref() == Some("WIN")).count();
                let losses = trades.iter().filter(|t| t.outcome.as_deref() == Some("LOSS")).count();
                let total_pnl: Decimal = trades.iter()
                    .filter(|t| t.outcome.is_some())
                    .map(|t| t.pnl_r.unwrap_or_default())
                    .sum();
                
                let win_rate = if completed_count > 0 {
                    wins as f64 / completed_count as f64 * 100.0
                } else { 0.0 };

                // T5.1: Build trade records for advanced metrics
                let trade_records: Vec<TradeRecord> = trades.iter()
                    .filter(|t| t.outcome.is_some())
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
                
                // T5.1: Calculate advanced metrics
                let advanced_metrics = AdvancedMetrics::calculate(&trade_records);
                
                // T5.2: Generate regime report
                let regime_report = RegimeReport::generate(&trade_records);

                // Now we can move trades into result
                let total_trades_count = trades.len();
                let result = BacktestResult {
                    symbol: symbol.clone(),
                    timeframe: interval.clone(),
                    total_trades: total_trades_count,
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
                info!("📊 Rapor: {} {} -> PnL: {}R (%{:.1} WR, {} Trades)", 
                      symbol, interval, total_pnl, win_rate, completed_count);
                info!("   📈 Expectancy: {:.3}R | PF: {:.2} | Sharpe: {:.2}", 
                      advanced_metrics.expectancy_r, 
                      advanced_metrics.profit_factor,
                      advanced_metrics.sharpe_ratio_approx);
                info!("   📉 Max Consec Loss: {} | Avg Duration: {:.1} candles", 
                      advanced_metrics.max_consecutive_losses,
                      advanced_metrics.avg_trade_duration_candles);
                
                // Track for summary
                let pair_key = format!("{} {}", symbol, interval);
                let pair_result = PairResult {
                    symbol: symbol.clone(),
                    timeframe: interval.clone(),
                    trades: completed_count,
                    wins,
                    losses,
                    win_rate,
                    pnl_r: total_pnl,
                    expectancy: advanced_metrics.expectancy_r,
                    profit_factor: advanced_metrics.profit_factor,
                    sharpe: advanced_metrics.sharpe_ratio_approx,
                    max_consec_loss: advanced_metrics.max_consecutive_losses,
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
                summary.total_trades += completed_count;
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
        summary.overall_win_rate = summary.total_wins as f64 / summary.total_trades as f64 * 100.0;
        summary.overall_expectancy = summary.overall_pnl_r.to_f64().unwrap_or(0.0) / summary.total_trades as f64;
        
        let gross_wins: f64 = summary.results_by_pair.iter()
            .filter(|r| r.pnl_r > Decimal::ZERO)
            .map(|r| r.pnl_r.to_f64().unwrap_or(0.0))
            .sum();
        let gross_losses: f64 = summary.results_by_pair.iter()
            .filter(|r| r.pnl_r < Decimal::ZERO)
            .map(|r| r.pnl_r.to_f64().unwrap_or(0.0).abs())
            .sum();
        summary.overall_profit_factor = if gross_losses > 0.0 { gross_wins / gross_losses } else { gross_wins };
    }
    
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
    info!("   Total Trades: {}", summary.total_trades);
    info!("   Wins: {} | Losses: {}", summary.total_wins, summary.total_losses);
    info!("   Win Rate: {:.1}%", summary.overall_win_rate);
    info!("   Total PnL: {}R", summary.overall_pnl_r);
    info!("   Expectancy: {:.3}R per trade", summary.overall_expectancy);
    info!("   Profit Factor: {:.2}", summary.overall_profit_factor);
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
    info!("   Signals Generated: {} ({:.2}% signal rate)", 
          bs.total_signals_generated, bs.signal_rate());
    info!("   Total Blocks: {}", bs.total_blocks());
    info!("");
    info!("   📍 Block Breakdown:");
    info!("      Wick Trap:        {:>6} blocks", bs.wick_trap_blocks);
    info!("      Flat EMA:         {:>6} blocks", bs.flat_ema_blocks);
    info!("      Low ATR:          {:>6} blocks", bs.low_atr_blocks);
    info!("      Bootstrap:        {:>6} blocks", bs.bootstrap_incomplete);
    info!("      Open Trade:       {:>6} blocks (legacy single-position)", bs.open_trade_blocks);
    info!("      Cooldown:         {:>6} blocks (post-close cooldown)", bs.cooldown_blocks);
    info!("      Score Too Low:    {:>6} blocks", bs.score_too_low);
    info!("      Policy Blocked:   {:>6} blocks", bs.policy_blocked);
    info!("");
    info!("   📍 Multi-Position Blocks (Phase 7):");
    info!("      Max Trades:       {:>6} blocks (reached max concurrent)", bs.max_trades_reached);
    info!("      Duplicate Ctx:    {:>6} blocks (same context_id)", bs.duplicate_context);
    info!("      Hedge Blocked:    {:>6} blocks (opposite direction)", bs.hedge_blocked);
    info!("      Context CD:       {:>6} blocks (context-specific cooldown)", bs.context_cooldown_blocks);
    info!("");
    info!("   📍 Phase 8 Blocks:");
    info!("      Trend Saturation: {:>6} blocks (slope weakening)", bs.trend_saturation_blocks);
    info!("      Weak Replaced:    {:>6} trades (replaced by stronger)", bs.weak_trade_replaced);
    info!("");
    info!("   📍 Phase 9 Exit Stats:");
    info!("      Max Duration:     {:>6} exits (forced after 14 candles)", bs.max_duration_exits);
    info!("      BE Applied:       {:>6} trades (moved SL to entry)", bs.be_applied_count);
    info!("      Partial TP:       {:>6} times (50% closed at 1R)", bs.partial_tp_count);
    info!("");
    info!("   📍 Phase 10 Safety:");
    info!("      Kill Switch:      {:>6} triggers (7 consec losses)", bs.kill_switch_triggered);
    info!("");
    
    // Results by pair table
    info!("📋 RESULTS BY PAIR");
    info!("───────────────────────────────────────────────────────────────");
    info!("   {:12} {:>6} {:>6} {:>7} {:>8} {:>8} {:>7}", 
          "PAIR", "TRADES", "WR%", "PnL(R)", "EXPECT", "PF", "SHARPE");
    info!("   {:─<12} {:─>6} {:─>6} {:─>7} {:─>8} {:─>8} {:─>7}", 
          "", "", "", "", "", "", "");
    
    for r in &summary.results_by_pair {
        let pair_name = format!("{} {}", 
            r.symbol.replace("USDT", ""), 
            r.timeframe);
        info!("   {:12} {:>6} {:>5.1}% {:>7} {:>7.3}R {:>7.2} {:>7.2}", 
              pair_name, 
              r.trades, 
              r.win_rate, 
              r.pnl_r, 
              r.expectancy, 
              r.profit_factor,
              r.sharpe);
    }
    
    info!("");
    info!("═══════════════════════════════════════════════════════════════");
    info!("                         🏁 END REPORT                         ");
    info!("═══════════════════════════════════════════════════════════════");
    info!("");
}

fn calculate_sl_tp(signal: &TradeSignal, ctx: &SymbolContext, entry: Decimal) -> (Decimal, Decimal) {
    let rr = Decimal::from_f64(1.5).unwrap();
    
    match signal.signal {
        SignalType::LONG => {
            // SL = Last Swing Low. Eğer yoksa %1 altı.
            let sl = ctx.structure.last_pivot_low.unwrap_or(entry * Decimal::from_f64(0.99).unwrap());
            // Koruma: Çok yakın SL varsa minimum %0.2 mesafe koy
            let safe_sl = if (entry - sl) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(0.995).unwrap()
            } else {
                sl
            };
            
            let risk = entry - safe_sl;
            let tp = entry + (risk * rr);
            (safe_sl, tp)
        },
        SignalType::SHORT => {
            let sl = ctx.structure.last_pivot_high.unwrap_or(entry * Decimal::from_f64(1.01).unwrap());
             let safe_sl = if (sl - entry) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(1.005).unwrap()
            } else {
                sl
            };
            
            let risk = safe_sl - entry;
            let tp = entry - (risk * rr);
            (safe_sl, tp)
        }
    }
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
    output_dir: &str
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
    
    info!("📅 Date Range: {} to {}", 
          candles.first().unwrap().open_time.format("%Y-%m-%d"),
          candles.last().unwrap().open_time.format("%Y-%m-%d"));
    
    let mut engine = SignalEngine::new_backtest_mode(); // T1.3: Backtest mode with shorter cooldowns
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
            info!("   ⏳ Progress: {:.0}% ({}/{} candles)", progress, candle_idx, total_candles);
        }
        
        // 1. Add candle
        ctx.add_candle(candle.clone());
        
        // T11.2: Try to reset kill switch if conditions are met
        let current_ema50_slope = Some(ctx.get_ema50_slope());
        let current_atr = ctx.atr_14.current_value;
        let median_atr = Some(ctx.get_median_atr_ratio() * current_atr.unwrap_or(Decimal::ONE)); // Approximate
        engine.try_reset_kill_switch(symbol, timeframe, candle_idx, current_ema50_slope, current_atr, median_atr);
        
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
        
        for trade in trades.iter_mut() {
            if trade.outcome.is_some() { continue; }
            
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
            
            // T9.2: Time-based BE - Move SL to entry after be_threshold_candles if profit < be_min_profit_r
            if !just_closed && !trade.is_be_applied && current_duration >= pool_config.be_threshold_candles {
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
            
            // Normal SL/TP checks (if not already closed by max duration)
            if !just_closed {
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
                    },
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
                let is_win = trade.outcome.as_deref() == Some("WIN");
                let ema50_slope = Some(ctx.get_ema50_slope());
                let current_atr = ctx.atr_14.current_value;
                engine.record_trade_result(symbol, timeframe, is_win, candle_idx, ema50_slope, current_atr);
                
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
    let completed_trades: Vec<&SimulatedTrade> = trades.iter()
        .filter(|t| t.outcome.is_some())
        .collect();
    
    let completed_count = completed_trades.len();
    let wins = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("WIN")).count();
    let losses = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("LOSS")).count();
    let be_count = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("BE")).count();
    let max_dur_count = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("MAX_DURATION")).count();
    // Win rate should only count WIN/LOSS, not BE or MAX_DURATION
    let decisive_trades = wins + losses;
    let win_rate = if decisive_trades > 0 { (wins as f64 / decisive_trades as f64) * 100.0 } else { 0.0 };
    
    let total_pnl: Decimal = completed_trades.iter()
        .filter_map(|t| t.pnl_r)
        .sum();
    
    let expectancy = if completed_count > 0 {
        total_pnl.to_f64().unwrap_or(0.0) / completed_count as f64
    } else {
        0.0
    };
    
    // T12.2: FIXED Profit Factor = sum(positive_R) / abs(sum(negative_R))
    // NOT (wins * 1.5) / losses - that's WRONG!
    let positive_r_sum: Decimal = completed_trades.iter()
        .filter_map(|t| t.pnl_r)
        .filter(|r| *r > Decimal::ZERO)
        .sum();
    let negative_r_sum: Decimal = completed_trades.iter()
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
    let returns: Vec<f64> = completed_trades.iter()
        .filter_map(|t| t.pnl_r.map(|r| r.to_f64().unwrap_or(0.0)))
        .collect();
    
    let sharpe = if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 { mean / std_dev } else { 0.0 }
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
    let durations: Vec<u32> = completed_trades.iter()
        .filter_map(|t| t.duration_candles)
        .collect();
    let avg_duration = if !durations.is_empty() {
        durations.iter().sum::<u32>() as f64 / durations.len() as f64
    } else {
        0.0
    };
    
    // T12.3: Sanity Check Guard - warn if metrics are inconsistent
    if profit_factor < 1.0 && total_pnl > Decimal::ZERO {
        warn!("⚠️ METRIC INCONSISTENCY: PF={:.2} < 1 but PnL={}R > 0. Check R accounting!", profit_factor, total_pnl);
    }
    if profit_factor > 1.0 && total_pnl < Decimal::ZERO {
        warn!("⚠️ METRIC INCONSISTENCY: PF={:.2} > 1 but PnL={}R < 0. Check R accounting!", profit_factor, total_pnl);
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
    info!("   Completed Trades: {} (Wins: {} | Losses: {} | BE: {} | MaxDur: {})", 
          completed_count, wins, losses, be_count, max_dur_count);
    info!("   Win Rate: {:.1}% (of decisive trades: {})", win_rate, decisive_trades);
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
    info!("   Signals Generated: {} ({:.2}% signal rate)", 
          block_stats.total_signals_generated,
          block_stats.signal_rate());
    info!("   Total Blocks: {}", block_stats.total_blocks());
    info!("");
    info!("   📊 Block Breakdown:");
    info!("      Wick Trap:        {:>6} blocks", block_stats.wick_trap_blocks);
    info!("      Flat EMA:         {:>6} blocks", block_stats.flat_ema_blocks);
    info!("      Low ATR:          {:>6} blocks", block_stats.low_atr_blocks);
    info!("      Bootstrap:        {:>6} blocks", block_stats.bootstrap_incomplete);
    info!("      Open Trade:       {:>6} blocks (legacy single-position)", block_stats.open_trade_blocks);
    info!("      Cooldown:         {:>6} blocks (post-close cooldown)", block_stats.cooldown_blocks);
    info!("      Score Too Low:    {:>6} blocks", block_stats.score_too_low);
    info!("      Policy Blocked:   {:>6} blocks", block_stats.policy_blocked);
    info!("");
    info!("   📊 Multi-Position Blocks:");
    info!("      Max Trades:       {:>6} blocks", block_stats.max_trades_reached);
    info!("      Duplicate Ctx:    {:>6} blocks", block_stats.duplicate_context);
    info!("      Hedge Blocked:    {:>6} blocks", block_stats.hedge_blocked);
    info!("      Context CD:       {:>6} blocks", block_stats.context_cooldown_blocks);
    info!("");
    
    // Multi-position metrics
    let pool = engine.get_position_pool();
    info!("   📊 Multi-Position Metrics:");
    info!("      Max Concurrent:   {:>6} trades", pool.max_concurrent_trades());
    info!("      Avg Concurrent:   {:>6.2} trades", pool.avg_concurrent_trades());
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
