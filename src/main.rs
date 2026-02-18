mod analytics;
mod backtest;
mod connect;
mod engine;
mod indicators;
mod ml_filter;
mod order_block;
mod policy;
mod state;
mod types;
mod alpaca;
mod paper_trader;

use crate::engine::{LstmMode, SignalEngine};
use crate::ml_filter::LstmFilter;
use crate::state::SymbolContext;
use crate::types::WsStreamMessage;
use config::Config;
use futures_util::StreamExt;
use rust_decimal::prelude::FromPrimitive;
use serde::Deserialize;
use std::collections::HashMap;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use dotenv::dotenv;

#[derive(Debug, Deserialize)]
struct AppSettings {
    app: AppConfig,
    trading: TradingConfig,
    backtest: Option<BacktestConfig>,
    #[serde(default)]
    binance: connect::BinanceSettings,
    #[serde(default)]
    ml: Option<MlConfig>,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(default = "default_mode")]
    mode: String,
    bootstrap_limit: usize,
    #[serde(default)]
    auto_reconnect: bool,
}

fn default_mode() -> String {
    "live".to_string()
}

#[derive(Debug, Deserialize)]
struct BacktestConfig {
    output_dir: String,
    days: i64,
    #[serde(default)]
    csv_file: Option<String>,
    #[serde(default = "default_csv_symbol")]
    csv_symbol: String,
    #[serde(default = "default_csv_timeframe")]
    csv_timeframe: String,
    #[serde(default = "default_exit_mode")]
    exit_mode: String,
    #[serde(default = "default_send_alpaca_signals")]
    send_alpaca_signals: bool,
    #[serde(default)]
    pool: Option<BacktestPoolConfig>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BacktestPoolConfig {
    #[serde(default)]
    be_threshold_candles: Option<u32>,
    #[serde(default)]
    be_min_profit_r: Option<f64>,
}

fn default_csv_symbol() -> String {
    "BTCUSDT".to_string()
}
fn default_csv_timeframe() -> String {
    "1m".to_string()
}

fn default_exit_mode() -> String {
    "supertrend".to_string()
}

fn default_send_alpaca_signals() -> bool {
    false
}

#[derive(Debug, Deserialize)]
struct TradingConfig {
    symbols: Vec<String>,
    timeframes: Vec<String>,
    #[serde(default)]
    execute_trades: bool,
    #[serde(default)]
    use_paper_trader: bool,
    #[serde(default = "default_paper_balance")]
    paper_initial_balance: f64,
    #[serde(default = "default_paper_state_file")]
    paper_state_file: String,
}

fn default_paper_balance() -> f64 {
    100000.0 // Default $100k
}

fn default_paper_state_file() -> String {
    "paper_trader_state.json".to_string()
}

#[derive(Debug, Deserialize, Clone)]
struct MlConfig {
    #[serde(default)]
    enabled: bool,
    model_path: String,
    meta_path: String,
    #[serde(default)]
    onnxruntime_path: Option<String>,
    #[serde(default)]
    mode: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if it exists
    dotenv().ok();

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
    info!("Binance market: {}", conf.binance.market_name());

    info!(
        "Ayarlar yüklendi: {} pairs izleniyor. MOD: {}",
        conf.trading.symbols.len(),
        conf.app.mode
    );

    if conf.app.mode == "backtest" {
        let lstm_filter = load_lstm_filter(&conf.ml);
        let lstm_mode = load_lstm_mode(&conf.ml);
        if let Some(bt_conf) = conf.backtest {
            let pool_overrides =
                bt_conf
                    .pool
                    .as_ref()
                    .map(|pool| backtest::runner::PoolConfigOverrides {
                        be_threshold_candles: pool.be_threshold_candles,
                        be_min_profit_r: pool.be_min_profit_r,
                    });
            // Check if CSV file is specified for local backtest
            if let Some(csv_file) = bt_conf.csv_file {
                info!("🗂️  Local CSV backtest mode");
                return backtest::runner::run_csv_backtest(
                    &csv_file,
                    &bt_conf.csv_symbol,
                    &bt_conf.csv_timeframe,
                    &bt_conf.exit_mode,
                    bt_conf.send_alpaca_signals,
                    &bt_conf.output_dir,
                    lstm_filter.clone(),
                    lstm_mode,
                    pool_overrides,
                )
                .await;
            }

            // Default: API-based backtest
            return backtest::runner::run_backtest(
                &conf.trading.symbols,
                &conf.trading.timeframes,
                bt_conf.days,
                &bt_conf.exit_mode,
                bt_conf.send_alpaca_signals,
                &bt_conf.output_dir,
                &conf.binance,
                lstm_filter.clone(),
                lstm_mode,
                pool_overrides,
            )
            .await;
        } else {
            error!("Backtest modu seçildi ama [backtest] konfigürasyonu eksik.");
            return Ok(());
        }
    }

    // Live Mode devamı...
    // Init Engine & State
    let mut engine = SignalEngine::new();
    if let Some(filter) = load_lstm_filter(&conf.ml) {
        engine.set_lstm_filter(filter);
    }
    engine.set_lstm_mode(load_lstm_mode(&conf.ml));
    let mut contexts: HashMap<String, SymbolContext> = HashMap::new();
    let client = connect::BinanceClient::with_settings(&conf.binance);

    // Initialize Alpaca client if execute_trades is enabled
    let alpaca_client = if conf.trading.execute_trades && !conf.trading.use_paper_trader {
        match alpaca::AlpacaClient::new() {
            Ok(client) => {
                info!("🦙 Alpaca client initialized for live trading");
                Some(client)
            }
            Err(e) => {
                error!("⚠️ Failed to initialize Alpaca client: {}", e);
                error!("   Trading execution will be disabled");
                None
            }
        }
    } else if conf.trading.use_paper_trader {
        None // Paper trader will be used instead
    } else {
        info!("ℹ️  Trade execution disabled (execute_trades = false)");
        None
    };

    // Initialize Paper Trader if enabled
    let mut paper_trader = if conf.trading.use_paper_trader && conf.trading.execute_trades {
        let state_path = std::path::Path::new(&conf.trading.paper_state_file);
        let trader = if state_path.exists() {
            match paper_trader::PaperTrader::load_from_file(state_path) {
                Ok(t) => {
                    info!("📊 Paper trader state loaded from: {}", conf.trading.paper_state_file);
                    t
                }
                Err(e) => {
                    warn!("⚠️ Failed to load paper trader state: {}", e);
                    warn!("   Starting with fresh state");
                    paper_trader::PaperTrader::new(
                        rust_decimal::Decimal::from_f64(conf.trading.paper_initial_balance).unwrap()
                    )
                }
            }
        } else {
            info!("💼 Paper trader initialized with ${} balance", conf.trading.paper_initial_balance);
            paper_trader::PaperTrader::new(
                rust_decimal::Decimal::from_f64(conf.trading.paper_initial_balance).unwrap()
            )
        };
        Some(trader)
    } else {
        None
    };

    // ... rest of live logic
    // Bootstrap (Historical Data)
    info!(
        "Bootstrap işlemi başlıyor ({:?} mum)...",
        conf.app.bootstrap_limit
    );

    for symbol in &conf.trading.symbols {
        // Her symbol için sadece belirtilen timeframe'leri yükle
        // Not: Şu an logic tek bir timeframe gibi varsayıyor olabilir, ama SymbolContext yapı olarak her interval için ayrı olmalı.
        // SymbolContext'i (Symbol, Interval) key ile saklamalıyız.
        // Ancak mevcut SymbolContext yapısı sadece symbol alıyor.
        // Basitlik için sadece ilk interval'ı veya config'deki her interval için key: "SYMBOL_INTERVAL" kullanalım.

        for interval in &conf.trading.timeframes {
            let key = format!("{}_{}", symbol, interval);
            info!("Fetching history for: {}", key);

            match client
                .fetch_candles(symbol, interval, conf.app.bootstrap_limit)
                .await
            {
                Ok(candles) => {
                    let mut ctx = SymbolContext::new(symbol.clone(), interval.clone());
                    // Context interval bilgisini de tutmalı mı? Şimdilik key üzerinden yönetiyoruz.
                    for c in candles {
                        ctx.add_candle(c);
                    }
                    contexts.insert(key.clone(), ctx);
                    info!(
                        "loaded {} candles for {}",
                        contexts[&key].candles.len(),
                        key
                    );
                }
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
        match connect::connect_stream(&conf.trading.symbols, &conf.trading.timeframes, &conf.binance).await {
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

                                                    // Update paper trader positions on every candle close
                                                    if let Some(ref mut trader) = paper_trader {
                                                        let current_close = ctx.candles.back()
                                                            .map(|c| c.close)
                                                            .unwrap_or_default();
                                                        if let Err(e) = trader.update_positions(&k.symbol, current_close) {
                                                            error!("Failed to update paper trader positions: {}", e);
                                                        }
                                                    }

                                                    // Sinyal Değerlendir
                                                    if let Some(signal) = engine.evaluate(ctx) {
                                                        // stdout -> pipe
                                                        println!(
                                                            "{}",
                                                            serde_json::to_string(&signal).unwrap()
                                                        );

                                                        // Execute trade with Paper Trader
                                                        if let Some(ref mut trader) = paper_trader {
                                                            info!("📊 Signal generated: {} {} @ ${}", 
                                                                  signal.signal, signal.symbol, signal.price);

                                                            // Calculate confidence multiplier
                                                            use crate::analytics::ConfidenceTier;
                                                            let confidence_tier = ConfidenceTier::from_score(signal.confidence as i32);
                                                            let confidence_multiplier = rust_decimal::Decimal::from_f64(
                                                                confidence_tier.position_size_multiplier()
                                                            ).unwrap();

                                                            // Open position
                                                            if let Err(e) = trader.open_position(
                                                                &signal,
                                                                ctx,
                                                                rust_decimal::Decimal::from_f64(0.01).unwrap(), // 1% risk
                                                                confidence_multiplier,
                                                            ) {
                                                                error!("Failed to open position: {}", e);
                                                            }

                                                            // Print status every 10 trades
                                                            if trader.total_trades % 10 == 0 && trader.total_trades > 0 {
                                                                trader.print_status();
                                                            }

                                                            // Save state
                                                            let state_path = std::path::Path::new(&conf.trading.paper_state_file);
                                                            if let Err(e) = trader.save_to_file(state_path) {
                                                                error!("Failed to save paper trader state: {}", e);
                                                            }
                                                        }
                                                        // Execute trade with Alpaca
                                                        else if let Some(alpaca) = &alpaca_client {
                                                            info!("📊 Signal generated: {} {} @ ${}", 
                                                                  signal.signal, signal.symbol, signal.price);
                                                            
                                                            // ⚠️ Alpaca crypto only supports LONG (spot buy) - skip SHORT signals
                                                            if matches!(signal.signal, crate::types::SignalType::SHORT) {
                                                                warn!("⚠️ Alpaca crypto doesn't support SHORT positions - signal ignored");
                                                                warn!("   Use a margin/futures platform if you want to short crypto");
                                                                continue;
                                                            }
                                                            
                                                            let entry_price = signal.price;
                                                            
                                                            // Build a temporary order to get SL/TP for position sizing
                                                            let (_temp_order, sl_price, _tp_price) = alpaca::build_market_entry_order(&signal, ctx, rust_decimal::Decimal::ZERO);
                                                            
                                                            // Calculate dynamic position size
                                                            match alpaca.calculate_position_size(
                                                                entry_price,
                                                                sl_price,
                                                                signal.confidence,
                                                                None, // Use default 1% risk
                                                            ).await {
                                                                Ok(qty) => {
                                                                    // Build final market entry order with calculated qty
                                                                    let (entry_order, sl, tp) = alpaca::build_market_entry_order(&signal, ctx, qty);
                                                                    
                                                                    info!("🚀 Submitting market entry order:");
                                                                    info!("   Symbol: {}, Side: {:?}, Qty: {}", entry_order.symbol, entry_order.side, entry_order.qty);
                                                                    info!("   Entry: ${}, SL: ${}, TP: ${}", entry_price, sl, tp);
                                                                    
                                                                    // Step 1: Submit entry order with retry
                                                                    match alpaca.submit_order_with_retry(entry_order.clone()).await {
                                                                        Ok(response) => {
                                                                            info!("✅ Entry order placed successfully!");
                                                                            info!("   Order ID: {}", response.id);
                                                                            info!("   Status: {}", response.status);
                                                                            
                                                                            // Step 2: Submit OCO orders (SL/TP) after entry
                                                                            // Get opposite side for exit orders
                                                                            let exit_side = alpaca::get_exit_side(&entry_order.side);
                                                                            
                                                                            info!("🎯 Submitting OCO orders (SL/TP)...");
                                                                            match alpaca.submit_oco_orders(
                                                                                &entry_order.symbol,
                                                                                exit_side,
                                                                                qty,
                                                                                sl,
                                                                                tp,
                                                                            ).await {
                                                                                Ok(oco_response) => {
                                                                                    info!("✅ OCO orders placed successfully!");
                                                                                    info!("   OCO Order ID: {}", oco_response.id);
                                                                                }
                                                                                Err(e) => {
                                                                                    error!("⚠️ Entry filled but failed to submit OCO orders: {}", e);
                                                                                    error!("   Manual intervention may be required!");
                                                                                }
                                                                            }
                                                                        }
                                                                        Err(e) => {
                                                                            error!("❌ Failed to submit entry order: {}", e);
                                                                        }
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    error!("❌ Failed to calculate position size: {}", e);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => error!("Candle parse error: {}", e),
                                            }
                                        }
                                    }
                                }
                                Err(_e) => {
                                    // Keepalive veya diğer mesajlar olabilir
                                    // error!("JSON parse error: {}", e);
                                }
                            }
                        }
                        Ok(Message::Ping(_)) => {
                            // Pong otomatik dönebilir veya manuel dönülebilir
                        }
                        Err(e) => {
                            error!("WS Error: {}", e);
                            break; // Inner loop'tan çık, outer loop reconnect edecek
                        }
                        _ => {}
                    }
                }
                warn!("WebSocket akışı kapandı.");
            }
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

fn load_lstm_filter(config: &Option<MlConfig>) -> Option<LstmFilter> {
    let cfg = config.as_ref()?;
    if !cfg.enabled {
        return None;
    }
    match LstmFilter::load(
        &cfg.model_path,
        &cfg.meta_path,
        cfg.onnxruntime_path.as_deref(),
    ) {
        Ok(filter) => Some(filter),
        Err(err) => {
            warn!("LSTM filter load failed: {}", err);
            None
        }
    }
}

fn load_lstm_mode(config: &Option<MlConfig>) -> LstmMode {
    let Some(cfg) = config.as_ref() else {
        return LstmMode::Filter;
    };

    match cfg.mode.as_deref() {
        Some("lstm_only") => LstmMode::LstmOnly,
        _ => LstmMode::Filter,
    }
}
