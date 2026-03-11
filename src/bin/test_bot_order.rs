use anyhow::Result;
use chrono::Utc;
use dotenv::dotenv;
use reqwest::Client;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::env;

use ebu_algo::binance_trader::BinanceFuturesTrader;
use ebu_algo::state::SymbolContext;
use ebu_algo::types::{RegimeContext, SignalType, TradeSignal};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("Starting Normal Bot Signal Test on Binance Futures...");

    // 2. Load .env file
    dotenv().ok();

    // Testnet'i zorlamak en sağlıklısıdır. Eger zorlamak istemezsek env'den alır.
    // Ancak gercek hesapta test ediyorsaniz risk_usdt 5.0 (5 dolar risk) olacaktir.

    // 3. Initialize the Trader
    // Using 5.0 risk amount and 1x leverage to match your config
    let risk_amount = Decimal::from_f64(5.0).unwrap();
    let leverage = 1;
    let trader = BinanceFuturesTrader::new(risk_amount, leverage, "sl_tp".to_string(), 1, false, 0.4, 0.35, false)?;

    tracing::info!("Trader initialized successfully. Fetching current BTC price...");

    // 4. Fetch current real price for BTCUSDT using public API to formulate a valid order
    let client = Client::new();
    let price_url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT";
    let resp = client.get(price_url).send().await?;
    let price_json: serde_json::Value = resp.json().await?;
    let price_str = price_json["price"].as_str().expect("No price found");
    let current_price = rust_decimal::Decimal::from_str_exact(price_str)?;

    tracing::info!("Current BTCUSDT Price: {}", current_price);

    // 5. Mock Symbol Context
    let mut ctx = SymbolContext::new("BTCUSDT".to_string(), "1m".to_string());

    // Mock ATR so SL/TP is wide enough (e.g. 500 USDT)
    let fake_atr = rust_decimal::Decimal::from_f64(500.0).unwrap();
    ctx.atr_14.current_value = Some(fake_atr);

    // 6. Create a mock LONG Signal
    let signal = TradeSignal {
        signal_id: "test-signal-001".to_string(),
        engine_version: "2.0".to_string(),
        symbol: "BTCUSDT".to_string(),
        timeframe: "1m".to_string(),
        signal: SignalType::LONG,
        price: current_price,
        confidence: 85,
        confidence_tier: "high".to_string(),
        timestamp: Utc::now(),
        reasons: vec!["Test Command".to_string()],
        context_id: None,
        regime_context: None,
    };

    tracing::info!("Executing test signal (MARKET ENTRY -> STOP LOSS -> TAKE PROFIT)...");

    // 7. Execute the normal bot logic
    match trader.execute_signal(&signal, &ctx).await {
        Ok(_) => {
            tracing::info!("✅ SUCCESS: Normal Bot Signal Order completely placed without errors!");
        }
        Err(e) => {
            tracing::error!("❌ ERROR: {}", e);
        }
    }

    Ok(())
}
