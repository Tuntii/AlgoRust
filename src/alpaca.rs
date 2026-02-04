use anyhow::{Context, Result};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, error, warn};
use crate::types::{TradeSignal, SignalType, TakeProfitSpec, StopLossSpec};
use crate::state::SymbolContext;

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
        let api_secret = env::var("ALPACA_SECRET_KEY").context("ALPACA_SECRET_KEY must be set in .env")?;
        let base_url = env::var("ALPACA_BASE_URL").unwrap_or_else(|_| "https://paper-api.alpaca.markets".to_string());
        
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
        let resp = self.client
            .get(url)
            .headers(self.headers())
            .send()
            .await?;
            
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
        
        let resp = self.client
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
        info!("Order submitted successfully: ID: {}, Status: {}", response.id, response.status);
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
                    let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1));
                    warn!("⚠️ Order attempt {}/{} failed: {}. Retrying in {:?}...", 
                          attempt, MAX_RETRIES, e, backoff);
                    sleep(backoff).await;
                }
            }
        }
    }

    async fn submit_order_internal(&self, order: OrderRequest) -> Result<OrderResponse> {
        let url = self.base_url.join("/v2/orders")?;
        
        let resp = self.client
            .post(url)
            .headers(self.headers())
            .json(&order)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        
        let status = resp.status();
        
        // Retry on server errors (5xx) or rate limiting (429)
        if status.is_server_error() || status.as_u16() == 429 {
            let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Retryable error ({}): {}", status, error_text);
        }
        
        // Non-retryable client errors (4xx except 429)
        if !status.is_success() {
            let error_text = resp.text().await?;
            error!("Alpaca Order Error (non-retryable): {}", error_text);
            anyhow::bail!("Failed to submit order: {}", error_text);
        }
        
        let response: OrderResponse = resp.json().await?;
        info!("✅ Order submitted successfully: ID: {}, Status: {}", response.id, response.status);
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
        let confidence_multiplier = Decimal::from_f64(confidence_tier.position_size_multiplier())
            .unwrap_or(Decimal::ZERO);
        
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
        info!("   Risk Amount ({}%): ${}", risk_pct * Decimal::from(100), risk_amount);
        info!("   Entry: ${}, SL: ${}, Risk/Unit: ${}", entry_price, sl_price, risk_per_unit);
        info!("   Base Size: {}, Confidence Tier: {:?} ({}x)", base_position_size, confidence_tier, confidence_multiplier);
        info!("   Final Position Size: {}", result);
        
        Ok(result)
    }
}

/// Build a bracket order (OCO) with automatic SL/TP from a trade signal
/// 
/// # Arguments
/// * `signal` - The trade signal to convert into an order
/// * `ctx` - Symbol context for pivot-based SL/TP calculation
/// * `qty` - Position size in base currency
/// 
/// # Returns
/// OrderRequest with bracket order fields populated
pub fn build_bracket_order(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    qty: Decimal,
) -> OrderRequest {
    use rust_decimal::prelude::FromStr;

    let entry = signal.price;
    let (sl_price, tp_price) = calculate_sl_tp(signal, ctx, entry);

    // Convert Binance symbol to Alpaca format (BTCUSDT -> BTC/USD)
    let alpaca_symbol = convert_to_alpaca_symbol(&signal.symbol);

    let side = match signal.signal {
        SignalType::LONG => Side::Buy,
        SignalType::SHORT => Side::Sell,
    };

    OrderRequest {
        symbol: alpaca_symbol,
        qty,
        side,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Gtc,
        limit_price: None,
        stop_price: None,
        // Bracket order configuration
        order_class: Some("bracket".to_string()),
        take_profit: Some(TakeProfitSpec {
            limit_price: tp_price,
        }),
        stop_loss: Some(StopLossSpec {
            stop_price: sl_price,
            limit_price: None, // Use market stop
        }),
    }
}

/// Calculate SL/TP based on pivots and risk/reward
/// Extracted from backtest/runner.rs for reusability
fn calculate_sl_tp(signal: &TradeSignal, ctx: &SymbolContext, entry: Decimal) -> (Decimal, Decimal) {
    use rust_decimal::prelude::FromStr;

    let default_rr = Decimal::from_f64(1.5).unwrap();
    let min_profit_pct = Decimal::from_f64(0.005).unwrap(); // Min 0.5% profit target

    match signal.signal {
        SignalType::LONG => {
            // SL = Last Swing Low, or 1% below entry
            let sl = ctx.structure.last_pivot_low
                .unwrap_or(entry * Decimal::from_f64(0.99).unwrap());
            
            // Safety: Minimum 0.2% distance if SL too close
            let safe_sl = if (entry - sl) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(0.995).unwrap()
            } else {
                sl
            };
            
            let risk = entry - safe_sl;

            // TP STRATEGY: Pivot-Based Target
            // Target: Nearest Pivot High above entry
            let mut target_tp = None;
            let mut best_tp = Decimal::MAX;
            
            for &pivot in &ctx.pivot_high_history {
                // Pivot must be at least 0.5% above entry
                if pivot > entry * (Decimal::ONE + min_profit_pct) {
                    if pivot < best_tp {
                        best_tp = pivot;
                        target_tp = Some(pivot);
                    }
                }
            }
            
            let tp = target_tp.unwrap_or_else(|| entry + (risk * default_rr));
            (safe_sl, tp)
        },
        SignalType::SHORT => {
            // SL = Last Swing High, or 1% above entry
            let sl = ctx.structure.last_pivot_high
                .unwrap_or(entry * Decimal::from_f64(1.01).unwrap());
            
            let safe_sl = if (sl - entry) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(1.005).unwrap()
            } else {
                sl
            };
            
            let risk = safe_sl - entry;

            // TP STRATEGY: Pivot-Based Target
            // Target: Nearest Pivot Low below entry
            let mut target_tp = None;
            let mut best_tp = Decimal::MIN;

            for &pivot in &ctx.pivot_low_history {
                // Pivot must be at least 0.5% below entry
                if pivot < entry * (Decimal::ONE - min_profit_pct) {
                    if pivot > best_tp {
                        best_tp = pivot;
                        target_tp = Some(pivot);
                    }
                }
            }

            let tp = target_tp.unwrap_or_else(|| entry - (risk * default_rr));
            (safe_sl, tp)
        }
    }
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
