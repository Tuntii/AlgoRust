use crate::state::SymbolContext;
use crate::types::{SignalType, StopLossSpec, TradeSignal};
use anyhow::{Context, Result};
use reqwest::{Client, Url};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct AlpacaClient {
    client: Client,
    base_url: Url,
    api_key: String,
    api_secret: String,
}

#[derive(Debug, Serialize, Clone)]
pub enum Side {
    #[serde(rename = "buy")]
    Buy,
    #[serde(rename = "sell")]
    Sell,
}

#[derive(Debug, Serialize, Clone)]
pub enum OrderType {
    #[serde(rename = "market")]
    Market,
    #[serde(rename = "limit")]
    Limit,
}

#[derive(Debug, Serialize, Clone)]
pub enum TimeInForce {
    #[serde(rename = "day")]
    Day,
    #[serde(rename = "gtc")]
    Gtc,
    #[serde(rename = "ioc")]
    Ioc,
}

#[derive(Debug, Serialize, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub qty: Decimal,
    pub side: Side,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_price: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_price: Option<Decimal>,
    // Bracket order fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_class: Option<String>, // "bracket", "oto", "oco"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take_profit: Option<crate::types::TakeProfitSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss: Option<crate::types::StopLossSpec>,
}

#[derive(Debug, Deserialize)]
pub struct Account {
    pub id: String,
    pub account_number: String,
    pub status: String,
    pub currency: String,
    pub buying_power: String,
    pub cash: String,
    pub portfolio_value: String,
}

#[derive(Debug, Deserialize)]
pub struct OrderResponse {
    pub id: String,
    pub client_order_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub submitted_at: String,
    pub filled_at: Option<String>,
    pub expired_at: Option<String>,
    pub canceled_at: Option<String>,
    pub failed_at: Option<String>,
    pub asset_id: String,
    pub symbol: String,
    pub asset_class: String,
    pub notional: Option<String>,
    pub qty: Option<String>,
    pub filled_qty: String,
    pub filled_avg_price: Option<String>,
    pub order_class: String,
    pub order_type: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub side: String,
    pub time_in_force: String,
    pub limit_price: Option<String>,
    pub stop_price: Option<String>,
    pub status: String,
}

impl AlpacaClient {
    pub fn new() -> Result<Self> {
        let api_key = env::var("ALPACA_API_KEY").context("ALPACA_API_KEY must be set in .env")?;
        let api_secret =
            env::var("ALPACA_SECRET_KEY").context("ALPACA_SECRET_KEY must be set in .env")?;
        let base_url = env::var("ALPACA_BASE_URL")
            .unwrap_or_else(|_| "https://paper-api.alpaca.markets".to_string());

        info!("🦙 Alpaca API Bağlantısı Başlatılıyor: {}", base_url);

        Ok(Self {
            client: Client::new(),
            base_url: Url::parse(&base_url)?,
            api_key,
            api_secret,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", self.api_key.parse().unwrap());
        headers.insert("APCA-API-SECRET-KEY", self.api_secret.parse().unwrap());
        headers
    }

    pub async fn get_account(&self) -> Result<Account> {
        let url = self.base_url.join("/v2/account")?;
        let resp = self.client.get(url).headers(self.headers()).send().await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            error!("Alpaca Account Error: {}", error_text);
            anyhow::bail!("Failed to get account: {}", error_text);
        }

        let account: Account = resp.json().await?;
        Ok(account)
    }

    pub async fn submit_order(&self, order: OrderRequest) -> Result<OrderResponse> {
        let url = self.base_url.join("/v2/orders")?;
        info!("Submitting order: {:?}", order);

        let resp = self
            .client
            .post(url)
            .headers(self.headers())
            .json(&order)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            error!("Alpaca Order Error: {}", error_text);
            anyhow::bail!("Failed to submit order: {}", error_text);
        }

        let response: OrderResponse = resp.json().await?;
        info!(
            "Order submitted successfully: ID: {}, Status: {}",
            response.id, response.status
        );
        Ok(response)
    }

    /// Get order status by ID
    pub async fn get_order(&self, order_id: &str) -> Result<OrderResponse> {
        let url = self.base_url.join(&format!("/v2/orders/{}", order_id))?;

        let resp = self
            .client
            .get(url)
            .headers(self.headers())
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            error!("Failed to get order: {}", error_text);
            anyhow::bail!("Failed to get order: {}", error_text);
        }

        let response: OrderResponse = resp.json().await?;
        Ok(response)
    }

    /// Submit OCO orders (stop-loss and take-profit) for an existing position
    /// Alpaca crypto doesn't support bracket orders, so we submit SL/TP as separate OCO orders
    pub async fn submit_oco_orders(
        &self,
        symbol: &str,
        side: Side, // Opposite of entry side (sell for long, buy for short)
        qty: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
    ) -> Result<OrderResponse> {
        let url = self.base_url.join("/v2/orders")?;

        // Create OCO order: take_profit as limit, stop_loss as stop
        let oco_order = OrderRequest {
            symbol: symbol.to_string(),
            qty,
            side,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::Gtc,
            limit_price: Some(tp_price),
            stop_price: None,
            order_class: Some("oco".to_string()),
            take_profit: None,
            stop_loss: Some(StopLossSpec {
                stop_price: sl_price,
                limit_price: None,
            }),
        };

        info!(
            "Submitting OCO orders: TP @ ${}, SL @ ${}",
            tp_price, sl_price
        );

        let resp = self
            .client
            .post(url)
            .headers(self.headers())
            .json(&oco_order)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            error!("Failed to submit OCO orders: {}", error_text);
            anyhow::bail!("Failed to submit OCO orders: {}", error_text);
        }

        let response: OrderResponse = resp.json().await?;
        info!("✅ OCO orders submitted successfully");
        Ok(response)
    }

    /// Submit order with exponential backoff retry logic
    /// Retries on 5xx server errors and 429 rate limiting
    pub async fn submit_order_with_retry(&self, order: OrderRequest) -> Result<OrderResponse> {
        const MAX_RETRIES: u32 = 3;
        const INITIAL_BACKOFF_MS: u64 = 1000;

        let mut attempt = 0;

        loop {
            attempt += 1;

            match self.submit_order_internal(order.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) if attempt >= MAX_RETRIES => {
                    error!("❌ Order failed after {} attempts: {}", MAX_RETRIES, e);
                    return Err(e);
                }
                Err(e) => {
                    let backoff =
                        Duration::from_millis(INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1));
                    warn!(
                        "⚠️ Order attempt {}/{} failed: {}. Retrying in {:?}...",
                        attempt, MAX_RETRIES, e, backoff
                    );
                    sleep(backoff).await;
                }
            }
        }
    }

    async fn submit_order_internal(&self, order: OrderRequest) -> Result<OrderResponse> {
        let url = self.base_url.join("/v2/orders")?;

        let resp = self
            .client
            .post(url)
            .headers(self.headers())
            .json(&order)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        let status = resp.status();

        // Retry on server errors (5xx) or rate limiting (429)
        if status.is_server_error() || status.as_u16() == 429 {
            let error_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Retryable error ({}): {}", status, error_text);
        }

        // Non-retryable client errors (4xx except 429)
        if !status.is_success() {
            let error_text = resp.text().await?;
            error!("Alpaca Order Error (non-retryable): {}", error_text);
            anyhow::bail!("Failed to submit order: {}", error_text);
        }

        let response: OrderResponse = resp.json().await?;
        info!(
            "✅ Order submitted successfully: ID: {}, Status: {}",
            response.id, response.status
        );
        Ok(response)
    }

    /// Calculate dynamic position size based on account balance and risk parameters
    ///
    /// # Arguments
    /// * `entry_price` - Entry price for the trade
    /// * `sl_price` - Stop loss price
    /// * `confidence` - Signal confidence (0-100)
    /// * `risk_percent` - Risk percentage of portfolio per trade (default: 0.01 = 1%)
    ///
    /// # Returns
    /// Position size in base currency (e.g., BTC amount for BTC/USD)
    pub async fn calculate_position_size(
        &self,
        entry_price: Decimal,
        sl_price: Decimal,
        confidence: u8,
        risk_percent: Option<Decimal>,
    ) -> Result<Decimal> {
        use crate::analytics::ConfidenceTier;
        use rust_decimal::prelude::FromStr;

        // Get account balance
        let account = self.get_account().await?;
        let portfolio_value = Decimal::from_str(&account.portfolio_value)
            .context("Failed to parse portfolio_value")?;

        // Use default 1% risk if not specified
        let risk_pct = risk_percent.unwrap_or_else(|| Decimal::from_str("0.01").unwrap());

        // Calculate risk amount in USD
        let risk_amount = portfolio_value * risk_pct;

        // Calculate risk per unit (distance from entry to SL)
        let risk_per_unit = (entry_price - sl_price).abs();

        // Prevent division by zero
        if risk_per_unit == Decimal::ZERO {
            warn!("⚠️ Risk per unit is zero, using minimum position size");
            return Ok(Decimal::from_str("0.001").unwrap());
        }

        // Base position size
        let base_position_size = risk_amount / risk_per_unit;

        // Apply confidence multiplier
        let confidence_tier = ConfidenceTier::from_score(confidence as i32);
        let confidence_multiplier =
            Decimal::from_f64(confidence_tier.position_size_multiplier()).unwrap_or(Decimal::ZERO);

        let final_position_size = base_position_size * confidence_multiplier;

        // Apply safety limits
        let max_position_value = portfolio_value * Decimal::from_str("0.10").unwrap(); // Max 10% of portfolio
        let position_value = final_position_size * entry_price;

        let safe_position_size = if position_value > max_position_value {
            warn!("⚠️ Position size exceeds 10% of portfolio, capping at max");
            max_position_value / entry_price
        } else {
            final_position_size
        };

        // Minimum position size check
        let min_position_size = Decimal::from_str("0.001").unwrap();
        let result = safe_position_size.max(min_position_size);

        info!("💰 Position Size Calculation:");
        info!("   Portfolio Value: ${}", portfolio_value);
        info!(
            "   Risk Amount ({}%): ${}",
            risk_pct * Decimal::from(100),
            risk_amount
        );
        info!(
            "   Entry: ${}, SL: ${}, Risk/Unit: ${}",
            entry_price, sl_price, risk_per_unit
        );
        info!(
            "   Base Size: {}, Confidence Tier: {:?} ({}x)",
            base_position_size, confidence_tier, confidence_multiplier
        );
        info!("   Final Position Size: {}", result);

        Ok(result)
    }
}

/// Build market entry order for crypto (without bracket)
/// Alpaca crypto doesn't support bracket orders, so we submit entry first,
/// then submit OCO orders (SL/TP) after entry is filled
///
/// # Arguments
/// * `signal` - The trade signal to convert into an order
/// * `ctx` - Symbol context for pivot-based SL/TP calculation
/// * `qty` - Position size in base currency
///
/// # Returns
/// OrderRequest for market entry (simple order) and (sl_price, tp_price) tuple
pub fn build_market_entry_order(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    qty: Decimal,
) -> (OrderRequest, Decimal, Decimal) {
    let entry = signal.price;
    let (sl_price, tp_price) = calculate_sl_tp(signal, ctx, entry);

    // Convert Binance symbol to Alpaca format (BTCUSDT -> BTC/USD)
    let alpaca_symbol = convert_to_alpaca_symbol(&signal.symbol);

    let side = match signal.signal {
        SignalType::LONG => Side::Buy,
        SignalType::SHORT => Side::Sell,
    };

    let order = OrderRequest {
        symbol: alpaca_symbol,
        qty,
        side,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Gtc,
        limit_price: None,
        stop_price: None,
        // No bracket/OCO for crypto - submit separately
        order_class: None,
        take_profit: None,
        stop_loss: None,
    };

    (order, sl_price, tp_price)
}

/// Get opposite side for exit orders (buy -> sell, sell -> buy)
pub fn get_exit_side(entry_side: &Side) -> Side {
    match entry_side {
        Side::Buy => Side::Sell,
        Side::Sell => Side::Buy,
    }
}

/// Calculate SL/TP based on Order Blocks (Smart Money TP/SL)
/// Fallback: pivot seviyeleri ve ATR tabanlı hesaplama
pub fn calculate_sl_tp_pub(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    entry: Decimal,
) -> (Decimal, Decimal) {
    let levels = ctx.calculate_trade_levels(&signal.signal, entry);
    (levels.sl, levels.tp2)
}

fn calculate_sl_tp(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    entry: Decimal,
) -> (Decimal, Decimal) {
    calculate_sl_tp_pub(signal, ctx, entry)
}

/// Convert Binance symbol format to Alpaca format
/// BTCUSDT -> BTC/USD
/// ETHUSDT -> ETH/USD
fn convert_to_alpaca_symbol(binance_symbol: &str) -> String {
    if binance_symbol.ends_with("USDT") {
        format!("{}/USD", binance_symbol.replace("USDT", ""))
    } else {
        binance_symbol.to_string()
    }
}
