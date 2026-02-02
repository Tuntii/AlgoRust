mod types;
mod indicators;
mod state;
mod engine;
mod connect;
mod backtest;

use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;
use crate::state::SymbolContext;
use crate::engine::SignalEngine;
use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::protocol::Message;
use crate::types::WsStreamMessage;

#[derive(Debug, Deserialize)]
struct AppSettings {
    app: AppConfig,
    trading: TradingConfig,
    backtest: Option<BacktestConfig>,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(default = "default_mode")]
    mode: String,
    bootstrap_limit: usize,
    #[serde(default)]
    auto_reconnect: bool,
}

fn default_mode() -> String { "live".to_string() }

#[derive(Debug, Deserialize)]
struct BacktestConfig {
    output_dir: String,
    days: i64,
}

#[derive(Debug, Deserialize)]
struct TradingConfig {
    symbols: Vec<String>,
    timeframes: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Loglamayı başlat (stderr)
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_writer(std::io::stderr) 
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Binance Price Action Engine başlatılıyor...");
    
    // Config Yükle
    let settings = Config::builder()
        .add_source(config::File::with_name("config"))
        .build()?;
    let conf: AppSettings = settings.try_deserialize()?;
    
    info!("Ayarlar yüklendi: {} pairs izleniyor. MOD: {}", conf.trading.symbols.len(), conf.app.mode);

    if conf.app.mode == "backtest" {
        if let Some(bt_conf) = conf.backtest {
            return backtest::runner::run_backtest(
                &conf.trading.symbols,
                &conf.trading.timeframes,
                bt_conf.days,
                &bt_conf.output_dir
            ).await;
        } else {
            error!("Backtest modu seçildi ama [backtest] konfigürasyonu eksik.");
            return Ok(());
        }
    }

    // Live Mode devamı...
    // Init Engine & State
    let engine = SignalEngine::new();
    let mut contexts: HashMap<String, SymbolContext> = HashMap::new();
    let client = connect::BinanceClient::new();
    
    // ... rest of live logic
    // Bootstrap (Historical Data)
    info!("Bootstrap işlemi başlıyor ({:?} mum)...", conf.app.bootstrap_limit);
    
    for symbol in &conf.trading.symbols {
        // Her symbol için sadece belirtilen timeframe'leri yükle
        // Not: Şu an logic tek bir timeframe gibi varsayıyor olabilir, ama SymbolContext yapı olarak her interval için ayrı olmalı.
        // SymbolContext'i (Symbol, Interval) key ile saklamalıyız.
        // Ancak mevcut SymbolContext yapısı sadece symbol alıyor.
        // Basitlik için sadece ilk interval'ı veya config'deki her interval için key: "SYMBOL_INTERVAL" kullanalım.
        
        for interval in &conf.trading.timeframes {
            let key = format!("{}_{}", symbol, interval);
            info!("Fetching history for: {}", key);
            
            match client.fetch_candles(symbol, interval, conf.app.bootstrap_limit).await {
                Ok(candles) => {
                    let mut ctx = SymbolContext::new(symbol.clone(), interval.clone());
                    // Context interval bilgisini de tutmalı mı? Şimdilik key üzerinden yönetiyoruz.
                    for c in candles {
                        ctx.add_candle(c);
                    }
                    contexts.insert(key.clone(), ctx);
                    info!("loaded {} candles for {}", contexts[&key].candles.len(), key);
                },
                Err(e) => {
                    error!("Bootstrap failed for {}: {}", key, e);
                    // Fail hard or continue? PRD says graceful.
                }
            }
        }
    }
    
    info!("Bootstrap tamamlandı. Sistem döngüsüne giriliyor...");
    
    loop {
        info!("WebSocket başlatılıyor...");
        match connect::connect_stream(&conf.trading.symbols, &conf.trading.timeframes).await {
            Ok(mut ws_stream) => {
                info!("✅ Bağlantı başarılı. Sinyaller bekleniyor...");

                // Event Loop
                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            // Parse
                            match serde_json::from_str::<WsStreamMessage>(&text) {
                                Ok(event) => {
                                    let k = event.data.kline;
                                    if k.is_closed {
                                        let key = format!("{}_{}", k.symbol, k.interval);
                                        
                                        if let Some(ctx) = contexts.get_mut(&key) {
                                            match k.to_candle() {
                                                Ok(candle) => {
                                                    ctx.add_candle(candle);
                                                    
                                                    // Sinyal Değerlendir
                                                    if let Some(signal) = engine.evaluate(ctx) {
                                                        // stdout -> pipe
                                                        println!("{}", serde_json::to_string(&signal).unwrap());
                                                    }
                                                },
                                                Err(e) => error!("Candle parse error: {}", e),
                                            }
                                        }
                                    }
                                },
                                Err(e) => {
                                    // Keepalive veya diğer mesajlar olabilir
                                    // error!("JSON parse error: {}", e);
                                }
                            }
                        },
                        Ok(Message::Ping(_)) => {
                            // Pong otomatik dönebilir veya manuel dönülebilir
                        },
                        Err(e) => {
                            error!("WS Error: {}", e);
                            break; // Inner loop'tan çık, outer loop reconnect edecek
                        },
                        _ => {}
                    }
                }
                warn!("WebSocket akışı kapandı.");
            },
            Err(e) => {
                error!("Bağlanırken hata oluştu: {}", e);
            }
        }

        if !conf.app.auto_reconnect {
            warn!("Auto-reconnect kapalı, çıkış yapılıyor.");
            break;
        }

        info!("5 saniye içinde yeniden bağlanılacak...");
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
    
    Ok(())
}
