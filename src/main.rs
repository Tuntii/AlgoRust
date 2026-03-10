mod alpaca;
mod analytics;
mod backtest;
mod binance_trader;
mod connect;
mod engine;
mod indicators;
mod ml_filter;
mod order_block;
mod paper_trader;
mod policy;
mod state;
mod types;

use crate::engine::{LstmMode, SignalEngine};
use crate::ml_filter::LstmFilter;
use crate::state::SymbolContext;
use crate::types::WsStreamMessage;
use config::Config;
use dotenv::dotenv;
use futures_util::StreamExt;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use serde::Deserialize;
use std::collections::HashMap;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{error, info, warn, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

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
    #[serde(default = "default_sltp_mode")]
    sltp_mode: String,
    #[serde(default = "default_indicator_profile")]
    indicator_profile: String,
    #[serde(default = "default_send_alpaca_signals")]
    send_alpaca_signals: bool,
    #[serde(default)]
    pool: Option<BacktestPoolConfig>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BacktestPoolConfig {
    #[serde(default)]
    max_trade_duration_candles: Option<u32>,
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

fn default_sltp_mode() -> String {
    "pivot".to_string()
}

fn default_indicator_profile() -> String {
    "baseline".to_string()
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
    #[serde(default = "default_leverage")]
    leverage: u32,
    #[serde(default = "default_risk_amount")]
    risk_amount_usdt: f64,
    #[serde(default = "default_live_exit_mode")]
    live_exit_mode: String,
    /// Aynı anda açık olabilecek maksimum pozisyon sayısı.
    /// Margin cap hesaplanırken bakiye bu sayıya bölünür.
    #[serde(default = "default_max_positions")]
    max_positions: u32,
    /// Trailing stop aktif mi? (indicator_flip modunda kâr koruma)
    #[serde(default)]
    trailing_stop_enabled: bool,
    /// Trailing stop callback oranı (% — 0.1-5.0)
    #[serde(default = "default_trailing_callback_rate")]
    trailing_callback_rate: f64,
    /// TP mesafesinin yüzde kaçında trailing stop aktif olacak (0.0-1.0)
    #[serde(default = "default_trailing_activation_pct")]
    trailing_activation_pct: f64,
}

fn default_max_positions() -> u32 {
    1
}

fn default_trailing_callback_rate() -> f64 {
    0.4
}

fn default_trailing_activation_pct() -> f64 {
    0.35
}

fn default_live_exit_mode() -> String {
    "sl_tp".to_string()
}

fn default_risk_amount() -> f64 {
    5.0
}

fn default_leverage() -> u32 {
    1
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

fn init_logging() -> anyhow::Result<()> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if it exists
    dotenv().ok();

    // Check for "test-order" CLI arg
    // Usage: cargo run -- test-order <SYMBOL> <AMOUNT>
    // Example: cargo run -- test-order BTCUSDT 5.0
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "test-order" {
        // Init logging (simplified)
        init_logging()?;

        let symbol = args.get(2).map(|s| s.as_str()).unwrap_or("BTCUSDT");
        let amount_f64 = args
            .get(3)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0);
        let amount = rust_decimal::Decimal::from_f64(amount_f64).unwrap_or_default();

        info!(
            "🧪 TEST MODE: Placing 1 market buy order on {} for ${}",
            symbol, amount
        );

        // Init trader (requires env vars to be present)
        let trader = match binance_trader::BinanceFuturesTrader::new(amount, 1, "sl_tp".to_string(), 1, false, 0.4, 0.35) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to init trader: {}", e);
                return Err(e);
            }
        };

        match trader.place_test_order(symbol, amount).await {
            Ok(id) => info!("✅ Test Order SUCCESS! Order ID: {}", id),
            Err(e) => error!("❌ Test Order FAILED: {:?}", e),
        }
        return Ok(());
    } else if args.len() > 1 && args[1] == "test-signal" {
        init_logging()?;

        let symbol = args.get(2).map(|s| s.as_str()).unwrap_or("BTCUSDT");
        let amount_f64 = args
            .get(3)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(5.0);
        let amount = rust_decimal::Decimal::from_f64(amount_f64).unwrap_or_default();

        info!(
            "🧪 TEST SIGNAL: Executing Normal Bot Signal on {} with ${} risk",
            symbol, amount
        );

        let trader = match binance_trader::BinanceFuturesTrader::new(amount, 1, "sl_tp".to_string(), 1, false, 0.4, 0.35) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to init trader: {}", e);
                return Err(e);
            }
        };

        // Fetch price
        let client = reqwest::Client::new();
        let price_url = format!(
            "https://fapi.binance.com/fapi/v1/ticker/price?symbol={}",
            symbol
        );
        let price_json: serde_json::Value = client.get(&price_url).send().await?.json().await?;
        let price_str = price_json["price"].as_str().unwrap_or("0");
        let current_price = rust_decimal::Decimal::from_str_exact(price_str).unwrap_or_default();

        let mut ctx = SymbolContext::new(symbol.to_string(), "1m".to_string());
        ctx.atr_14.current_value = Some(rust_decimal::Decimal::from_f64(100.0).unwrap());

        use chrono::Utc;
        let signal = crate::types::TradeSignal {
            signal_id: "test-signal-001".to_string(),
            engine_version: "2.0".to_string(),
            symbol: symbol.to_string(),
            timeframe: "1m".to_string(),
            signal: crate::types::SignalType::LONG,
            price: current_price,
            confidence: 90,
            confidence_tier: "high".to_string(),
            timestamp: Utc::now(),
            reasons: vec!["Test Signal Command".to_string()],
            context_id: None,
            regime_context: None,
        };

        match trader.execute_signal(&signal, &ctx).await {
            Ok(_result) => info!(
                "✅ TEST SIGNAL SUCCESS: Normal Bot Signal Order completely placed without errors!"
            ),
            Err(e) => error!("❌ TEST SIGNAL FAILED: {:?}", e),
        }
        return Ok(());
    }

    // Loglamayı başlat (stderr)
    init_logging()?;

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
                        max_trade_duration_candles: pool.max_trade_duration_candles,
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
                    &bt_conf.sltp_mode,
                    &bt_conf.indicator_profile,
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
                &bt_conf.sltp_mode,
                &bt_conf.indicator_profile,
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

    // Initialize Binance Futures trader if execute_trades=true and not paper
    let binance_trader = if conf.trading.execute_trades && !conf.trading.use_paper_trader {
        let risk_amount = rust_decimal::Decimal::from_f64(conf.trading.risk_amount_usdt)
            .unwrap_or(rust_decimal::Decimal::new(5, 0));
        match binance_trader::BinanceFuturesTrader::new(
            risk_amount,
            conf.trading.leverage,
            conf.trading.live_exit_mode.clone(),
            conf.trading.max_positions,
            conf.trading.trailing_stop_enabled,
            conf.trading.trailing_callback_rate,
            conf.trading.trailing_activation_pct,
        ) {
            Ok(trader) => {
                info!(
                    "💰 Binance Futures trader initialized (${} risk/trade, {}x leverage, max {} positions, trailing={})",
                    risk_amount, conf.trading.leverage, conf.trading.max_positions,
                    if conf.trading.trailing_stop_enabled { format!("{}% callback", conf.trading.trailing_callback_rate) } else { "off".to_string() }
                );
                Some(trader)
            }
            Err(e) => {
                error!("⚠️ Binance trader init failed: {}", e);
                error!("   Make sure BINANCE_API_KEY and BINANCE_API_SECRET are set in .env");
                None
            }
        }
    } else {
        None
    };
    // Alpaca kept for backwards compat but not used when binance_trader is active
    let _alpaca_client: Option<alpaca::AlpacaClient> = None;

    // Initialize Paper Trader if enabled
    let mut paper_trader = if conf.trading.use_paper_trader && conf.trading.execute_trades {
        let state_path = std::path::Path::new(&conf.trading.paper_state_file);
        let trader = if state_path.exists() {
            match paper_trader::PaperTrader::load_from_file(state_path) {
                Ok(t) => {
                    info!(
                        "📊 Paper trader state loaded from: {}",
                        conf.trading.paper_state_file
                    );
                    t
                }
                Err(e) => {
                    warn!("⚠️ Failed to load paper trader state: {}", e);
                    warn!("   Starting with fresh state");
                    paper_trader::PaperTrader::new(
                        rust_decimal::Decimal::from_f64(conf.trading.paper_initial_balance)
                            .unwrap(),
                    )
                }
            }
        } else {
            info!(
                "💼 Paper trader initialized with ${} balance",
                conf.trading.paper_initial_balance
            );
            paper_trader::PaperTrader::new(
                rust_decimal::Decimal::from_f64(conf.trading.paper_initial_balance).unwrap(),
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
            info!(
                "Fetching history for: {} ({} candles)",
                key, conf.app.bootstrap_limit
            );

            // Use paginated fetch when bootstrap_limit > 1000 (Binance per-request cap)
            let fetch_result = if conf.app.bootstrap_limit > 1000 {
                client
                    .fetch_candles_paginated(symbol, interval, conf.app.bootstrap_limit)
                    .await
            } else {
                client
                    .fetch_candles(symbol, interval, conf.app.bootstrap_limit)
                    .await
            };

            match fetch_result {
                Ok(candles) => {
                    let mut ctx = SymbolContext::new(symbol.clone(), interval.clone());
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
                }
            }
        }
    }

    info!("Bootstrap tamamlandı. Sistem döngüsüne giriliyor...");

    loop {
        info!("WebSocket başlatılıyor...");
        match connect::connect_stream(
            &conf.trading.symbols,
            &conf.trading.timeframes,
            &conf.binance,
        )
        .await
        {
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
                                                    // ─────────────────────────────────────────────
                                                    // ADIM 1: Önceki mumun sinyalini bu mumun
                                                    // açılış fiyatıyla execute et (next-bar open)
                                                    // ─────────────────────────────────────────────
                                                    if let Some(mut pending) = ctx.pending_signal.take() {
                                                        let planned_entry = pending.price;
                                                        pending.price = candle.open;
                                                        let entry_gap = pending.price - planned_entry;
                                                        let entry_gap_pct = if planned_entry.is_zero() {
                                                            Decimal::ZERO
                                                        } else {
                                                            (entry_gap / planned_entry) * Decimal::from(100)
                                                        };
                                                        info!(
                                                            "⏳ Pending sinyal execute ediliyor ({} {} @ ${}  — next-bar open)",
                                                            pending.signal, pending.symbol, pending.price
                                                        );
                                                        info!(
                                                            "📝 Pending signal detail [{} {}] ctx={} planned_close_entry={} next_open_entry={} gap={} ({}%) confidence={} tier={} reasons={:?}",
                                                            pending.symbol,
                                                            pending.timeframe,
                                                            pending.context_id.as_deref().unwrap_or("n/a"),
                                                            planned_entry,
                                                            pending.price,
                                                            entry_gap,
                                                            entry_gap_pct,
                                                            pending.confidence,
                                                            pending.confidence_tier,
                                                            pending.reasons,
                                                        );
                                                        info!(
                                                            "🧭 Live runtime snapshot before order [{} {}]: {}",
                                                            pending.symbol,
                                                            pending.timeframe,
                                                            ctx.live_diagnostic_snapshot(),
                                                        );
                                                        // stdout -> pipe
                                                        println!(
                                                            "{}",
                                                            serde_json::to_string(&pending).unwrap()
                                                        );

                                                        // Execute trade with Paper Trader
                                                        if let Some(ref mut trader) = paper_trader {
                                                            use crate::analytics::ConfidenceTier;
                                                            let confidence_tier =
                                                                ConfidenceTier::from_score(
                                                                    pending.confidence as i32,
                                                                );
                                                            let confidence_multiplier =
                                                                rust_decimal::Decimal::from_f64(
                                                                    confidence_tier
                                                                        .position_size_multiplier(),
                                                                )
                                                                .unwrap();
                                                            if let Err(e) = trader.open_position(
                                                                &pending,
                                                                ctx,
                                                                rust_decimal::Decimal::from_f64(0.01)
                                                                    .unwrap(),
                                                                confidence_multiplier,
                                                            ) {
                                                                error!("Failed to open position: {}", e);
                                                            }
                                                            if trader.total_trades % 10 == 0
                                                                && trader.total_trades > 0
                                                            {
                                                                trader.print_status();
                                                            }
                                                            let state_path = std::path::Path::new(
                                                                &conf.trading.paper_state_file,
                                                            );
                                                            if let Err(e) =
                                                                trader.save_to_file(state_path)
                                                            {
                                                                error!("Failed to save paper trader state: {}", e);
                                                            }
                                                        }
                                                        // Execute trade with Binance Futures
                                                        else if let Some(ref trader) = binance_trader {
                                                            match trader
                                                                .execute_signal(&pending, ctx)
                                                                .await
                                                            {
                                                                Ok(result) => {
                                                                    use crate::binance_trader::SignalExecResult;
                                                                    if result == SignalExecResult::Flipped {
                                                                        engine.record_trade_close(
                                                                            &pending.symbol,
                                                                            &pending.timeframe,
                                                                            ctx.total_candles_processed,
                                                                        );
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    error!("❌ Binance order failed: {}", e);
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // ─────────────────────────────────────────────
                                                    // ADIM 2: Mumu ekle ve pozisyonları güncelle
                                                    // ─────────────────────────────────────────────
                                                    ctx.add_candle(candle);

                                                    if let Some(ref mut trader) = paper_trader {
                                                        let current_close = ctx
                                                            .candles
                                                            .back()
                                                            .map(|c| c.close)
                                                            .unwrap_or_default();
                                                        if let Err(e) = trader.update_positions(
                                                            &k.symbol,
                                                            current_close,
                                                        ) {
                                                            error!("Failed to update paper trader positions: {}", e);
                                                        }
                                                    }

                                                    // ─────────────────────────────────────────────
                                                    // ADIM 3: Sinyal değerlendir → pending'e al
                                                    // (bir sonraki mum açılışında execute edilecek)
                                                    // ─────────────────────────────────────────────
                                                    if let Some(signal) = engine.evaluate(ctx) {
                                                        info!(
                                                            "📌 Sinyal tespit edildi, sonraki mum açılışı bekleniyor: {} {} @ ${}",
                                                            signal.signal, signal.symbol, signal.price
                                                        );
                                                        ctx.pending_signal = Some(signal);
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
