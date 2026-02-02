use anyhow::{Context, Result};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{info, error};

#[derive(Debug, Clone)]
pub struct AlpacaClient {
    client: Client,
    base_url: Url,
    api_key: String,
    api_secret: String,
}

#[derive(Debug, Serialize)]
pub enum Side {
    #[serde(rename = "buy")]
    Buy,
    #[serde(rename = "sell")]
    Sell,
}

#[derive(Debug, Serialize)]
pub enum OrderType {
    #[serde(rename = "market")]
    Market,
    #[serde(rename = "limit")]
    Limit,
}

#[derive(Debug, Serialize)]
pub enum TimeInForce {
    #[serde(rename = "day")]
    Day,
    #[serde(rename = "gtc")]
    Gtc,
    #[serde(rename = "ioc")]
    Ioc,
}

#[derive(Debug, Serialize)]
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
}
