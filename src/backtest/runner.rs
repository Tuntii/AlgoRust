use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Duration, Utc};
use std::fs::{self, File};
use std::io::Write;
use crate::types::{TradeSignal, Candle, SignalType, TrendState};
use crate::state::SymbolContext;
use crate::engine::SignalEngine;
use crate::connect::BinanceClient;
use tracing::{info, error};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct BacktestResult {
    symbol: String,
    timeframe: String,
    total_trades: usize,
    wins: usize,
    losses: usize,
    win_rate: f64,
    total_pnl_r: Decimal, // R-multiple cinsinden PnL
    signals: Vec<SimulatedTrade>,
}

#[derive(Serialize)]
struct SimulatedTrade {
    #[serde(flatten)]
    signal: TradeSignal,
    entry_price: Decimal,
    sl_price: Decimal,
    tp_price: Decimal,
    exit_price: Option<Decimal>,
    pnl_r: Option<Decimal>,
    outcome: Option<String>, // WIN, LOSS, OPEN
}

pub async fn run_backtest(
    symbols: &[String], 
    timeframes: &[String], 
    days: i64,
    output_dir: &str
) -> anyhow::Result<()> {
    info!("🔄 Backtest Başlatılıyor... (Son {} gün)", days);
    
    // Klasörü oluştur
    fs::create_dir_all(output_dir)?;
    
    let client = BinanceClient::new();
    let engine = SignalEngine::new();
    
    // Binance limit: 1000 candles per call. 
    // Basitlik için backtest şimdilik son 1000 mum ile sınırlı, 
    // ama pagination ile geriye gidilebilir.
    let limit = 1000; 

    for symbol in symbols {
        for interval in timeframes {
            info!("Testing {} {}...", symbol, interval);
            
            let mut ctx = SymbolContext::new(symbol.clone(), interval.clone());
            let mut trades: Vec<SimulatedTrade> = Vec::new();
            
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
                        if let Some(signal) = engine.evaluate(&ctx) {
                            // Basit SL/TP Stratejisi
                            // LONG: SL = Last Pivot Low, TP = 1.5R
                            // SHORT: SL = Last Pivot High, TP = 1.5R
                            
                            let entry = candle.close;
                            let (sl, tp) = calculate_sl_tp(&signal, &ctx, entry);
                            
                            trades.push(SimulatedTrade {
                                signal,
                                entry_price: entry,
                                sl_price: sl,
                                tp_price: tp,
                                exit_price: None,
                                pnl_r: None,
                                outcome: None,
                            });
                        }
                        
                         // 3. Açık pozisyonları yönet (Simülasyon)
                        for trade in trades.iter_mut() {
                            if trade.outcome.is_some() { continue; } // Zaten kapandı

                            // Trade sinyal mumuyla aynıysa skip et
                            if trade.signal.timestamp == candle.close_time.unwrap_or(candle.open_time) {
                                continue;
                            }
                            
                            // High/Low kontrolü
                            match trade.signal.signal {
                                SignalType::LONG => {
                                    if candle.low <= trade.sl_price {
                                        trade.outcome = Some("LOSS".to_string());
                                        trade.exit_price = Some(trade.sl_price);
                                        trade.pnl_r = Some(Decimal::from(-1));
                                    } else if candle.high >= trade.tp_price {
                                        trade.outcome = Some("WIN".to_string());
                                        trade.exit_price = Some(trade.tp_price);
                                        trade.pnl_r = Some(Decimal::from_f64(1.5).unwrap());
                                    }
                                },
                                SignalType::SHORT => {
                                    if candle.high >= trade.sl_price {
                                        trade.outcome = Some("LOSS".to_string());
                                        trade.exit_price = Some(trade.sl_price);
                                        trade.pnl_r = Some(Decimal::from(-1));
                                    } else if candle.low <= trade.tp_price {
                                        trade.outcome = Some("WIN".to_string());
                                        trade.exit_price = Some(trade.tp_price);
                                        trade.pnl_r = Some(Decimal::from_f64(1.5).unwrap());
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => error!("Data fetch failed: {}", e),
            }
            
            // Sonuçları Yazma
            if !trades.is_empty() {
                let completed_trades: Vec<&SimulatedTrade> = trades.iter().filter(|t| t.outcome.is_some()).collect();
                let wins = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("WIN")).count();
                let losses = completed_trades.iter().filter(|t| t.outcome.as_deref() == Some("LOSS")).count();
                let total_pnl: Decimal = completed_trades.iter().map(|t| t.pnl_r.unwrap_or_default()).sum();
                
                let win_rate = if !completed_trades.is_empty() {
                    wins as f64 / completed_trades.len() as f64 * 100.0
                } else { 0.0 };

                let result = BacktestResult {
                    symbol: symbol.clone(),
                    timeframe: interval.clone(),
                    total_trades: trades.len(),
                    wins,
                    losses,
                    win_rate,
                    total_pnl_r: total_pnl,
                    signals: trades,
                };

                let filename = format!("{}/{}_{}_backtest.json", output_dir, symbol, interval);
                let mut file = File::create(&filename)?;
                let json = serde_json::to_string_pretty(&result)?;
                file.write_all(json.as_bytes())?;
                
                 info!("📊 Rapor: {} {} -> PnL: {}R (%{:.1} WR, {} Trades)", 
                      symbol, interval, total_pnl, win_rate, result.total_trades);
            } else {
                info!("ℹ️  Sinyal bulunamadı: {} {}", symbol, interval);
            }
        }
    }
    
    info!("🏁 Backtest Tamamlandı.");
    Ok(())
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
