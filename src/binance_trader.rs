// ============================================================================
// BINANCE FUTURES TRADER
// Real-money order execution via Binance USDM Futures REST API
// Endpoints: POST /fapi/v1/order, GET /fapi/v2/balance
// Auth: HMAC-SHA256 signature on every signed request
// ============================================================================

use crate::state::SymbolContext;
use crate::types::{SignalType, TradeSignal};
use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::Sha256;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

type HmacSha256 = Hmac<Sha256>;

const FAPI_BASE: &str = "https://fapi.binance.com";
const TESTNET_BASE: &str = "https://testnet.binancefuture.com";
const DEFAULT_RECV_WINDOW_MS: u64 = 10_000;
const MAX_RECV_WINDOW_MS: u64 = 60_000;

pub struct BinanceFuturesTrader {
    client: Client,
    base_url: String,
    api_key: String,
    api_secret: String,
    recv_window_ms: u64,
    time_offset_ms: AtomicI64,
    time_offset_synced: AtomicBool,
    /// Risk fraction per trade (e.g. 0.01 = 1%)
    pub risk_fraction: Decimal,
}

impl BinanceFuturesTrader {
    /// Load credentials from .env:
    ///   BINANCE_API_KEY, BINANCE_API_SECRET
    ///   BINANCE_TESTNET=true  (optional, uses testnet if set)
    pub fn new(risk_fraction: Decimal) -> Result<Self> {
        let api_key = env::var("BINANCE_API_KEY").context("BINANCE_API_KEY must be set in .env")?;
        let api_secret =
            env::var("BINANCE_API_SECRET").context("BINANCE_API_SECRET must be set in .env")?;
        let testnet = env::var("BINANCE_TESTNET")
            .unwrap_or_default()
            .to_lowercase()
            == "true";
        let recv_window_ms = env::var("BINANCE_RECV_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v.clamp(1_000, MAX_RECV_WINDOW_MS))
            .unwrap_or(DEFAULT_RECV_WINDOW_MS);

        let base_url = if testnet {
            warn!("Binance TESTNET mode - no real money");
            TESTNET_BASE.to_string()
        } else {
            info!("Binance LIVE FUTURES mode");
            FAPI_BASE.to_string()
        };
        info!("Binance recvWindow={}ms", recv_window_ms);

        Ok(Self {
            client: Client::new(),
            base_url,
            api_key,
            api_secret,
            recv_window_ms,
            time_offset_ms: AtomicI64::new(0),
            time_offset_synced: AtomicBool::new(false),
            risk_fraction,
        })
    }

    fn local_timestamp_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    fn adjusted_timestamp_ms(&self) -> u64 {
        let local = Self::local_timestamp_ms();
        let offset = self.time_offset_ms.load(Ordering::Relaxed);
        local.saturating_add(offset).max(0) as u64
    }

    fn sign(&self, query: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC accepts any key size");
        mac.update(query.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn signed_url(&self, path: &str, mut params: Vec<(&str, String)>) -> String {
        params.push(("recvWindow", self.recv_window_ms.to_string()));
        params.push(("timestamp", self.adjusted_timestamp_ms().to_string()));

        let qs: String = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");
        let sig = self.sign(&qs);
        format!("{}{path}?{qs}&signature={sig}", self.base_url)
    }

    fn auth_headers(&self) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert("X-MBX-APIKEY", self.api_key.parse().unwrap());
        h
    }

    fn is_timestamp_error(body: &str) -> bool {
        body.contains("\"code\":-1021") || body.contains("recvWindow")
    }

    async fn sync_server_time(&self) -> Result<()> {
        let url = format!("{}/fapi/v1/time", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            let body = resp.text().await?;
            anyhow::bail!("time sync failed: {}", body);
        }

        let time: ServerTimeResponse = resp.json().await?;
        let local = Self::local_timestamp_ms();
        let offset = (time.server_time as i64).saturating_sub(local);
        self.time_offset_ms.store(offset, Ordering::Relaxed);
        self.time_offset_synced.store(true, Ordering::Relaxed);
        info!("Binance server time synced (offset={}ms)", offset);
        Ok(())
    }

    async fn send_signed_request(
        &self,
        method: reqwest::Method,
        path: &str,
        params: Vec<(&str, String)>,
    ) -> Result<reqwest::Response> {
        if !self.time_offset_synced.load(Ordering::Relaxed) {
            if let Err(err) = self.sync_server_time().await {
                warn!("Binance time sync failed; using local time: {}", err);
            }
        }

        let mut retried = false;
        loop {
            let url = self.signed_url(path, params.clone());
            let resp = self
                .client
                .request(method.clone(), &url)
                .headers(self.auth_headers())
                .send()
                .await?;

            if resp.status().is_success() {
                return Ok(resp);
            }

            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());

            if !retried && Self::is_timestamp_error(&body) {
                warn!("Binance rejected timestamp; syncing time and retrying once");
                self.sync_server_time().await?;
                retried = true;
                continue;
            }

            anyhow::bail!(
                "signed request failed ({} {}): {}",
                method.as_str(),
                status.as_u16(),
                body
            );
        }
    }

    pub async fn usdt_balance(&self) -> Result<Decimal> {
        let resp = self
            .send_signed_request(reqwest::Method::GET, "/fapi/v2/balance", vec![])
            .await
            .context("Balance fetch failed")?;
        let balances: Vec<BalanceEntry> = resp.json().await?;
        let usdt = balances
            .into_iter()
            .find(|b| b.asset == "USDT")
            .map(|b| b.available_balance)
            .unwrap_or(Decimal::ZERO);
        info!("Available USDT balance: ${}", usdt);
        Ok(usdt)
    }

    /// Contracts = (balance * risk_fraction) / |entry - sl|
    pub async fn position_qty(&self, entry: Decimal, sl: Decimal) -> Result<Decimal> {
        let balance = self.usdt_balance().await?;
        let risk_usd = balance * self.risk_fraction;
        let sl_distance = (entry - sl).abs();
        if sl_distance.is_zero() {
            warn!("SL distance is zero, using minimum qty 0.001");
            return Ok(Decimal::new(1, 3)); // 0.001
        }
        let qty = (risk_usd / sl_distance).round_dp(3);
        info!(
            "Size calc: balance=${} risk={}% risk_usd=${} sl_dist={} -> qty={}",
            balance,
            self.risk_fraction * Decimal::ONE_HUNDRED,
            risk_usd,
            sl_distance,
            qty
        );
        Ok(qty.max(Decimal::new(1, 3))) // enforce minimum 0.001
    }

    /// Set leverage for a symbol (call once per symbol before trading)
    pub async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<()> {
        let params = vec![
            ("symbol", symbol.to_string()),
            ("leverage", leverage.to_string()),
        ];
        self.send_signed_request(reqwest::Method::POST, "/fapi/v1/leverage", params)
            .await
            .with_context(|| format!("set_leverage failed for {}", symbol))?;
        info!("Leverage set to {}x for {}", leverage, symbol);
        Ok(())
    }

    /// Entry (MARKET) + SL (STOP_MARKET) + TP (TAKE_PROFIT_MARKET)
    pub async fn execute_signal(&self, signal: &TradeSignal, ctx: &SymbolContext) -> Result<()> {
        // Enforce 1x leverage before every order
        self.set_leverage(&signal.symbol, 1).await?;
        let entry = signal.price;
        let (sl, tp) = crate::alpaca::calculate_sl_tp_pub(signal, ctx, entry);

        let qty = self.position_qty(entry, sl).await?;

        let (side, sl_side, tp_side) = match signal.signal {
            SignalType::LONG => ("BUY", "SELL", "SELL"),
            SignalType::SHORT => ("SELL", "BUY", "BUY"),
        };

        // Precision: Binance requires specific decimal places per symbol
        let sl_str = format!("{:.2}", sl);
        let tp_str = format!("{:.2}", tp);
        let qty_str = format!("{:.3}", qty);

        info!(
            "Binance Futures signal: {} {} qty={} entry={} SL={} TP={}",
            side, signal.symbol, qty_str, entry, sl_str, tp_str
        );

        // 1. Market entry
        let entry_id = self
            .place_order(OrderParams {
                symbol: &signal.symbol,
                side,
                order_type: "MARKET",
                qty: &qty_str,
                price: None,
                stop_price: None,
                close_position: false,
            })
            .await?;
        info!("Market entry placed - orderId={}", entry_id);

        // 2. Stop-loss
        let sl_id = self
            .place_order(OrderParams {
                symbol: &signal.symbol,
                side: sl_side,
                order_type: "STOP_MARKET",
                qty: &qty_str,
                price: None,
                stop_price: Some(&sl_str),
                close_position: true,
            })
            .await?;
        info!("Stop-loss placed @ {} - orderId={}", sl_str, sl_id);

        // 3. Take-profit
        let tp_id = self
            .place_order(OrderParams {
                symbol: &signal.symbol,
                side: tp_side,
                order_type: "TAKE_PROFIT_MARKET",
                qty: &qty_str,
                price: None,
                stop_price: Some(&tp_str),
                close_position: true,
            })
            .await?;
        info!("Take-profit placed @ {} - orderId={}", tp_str, tp_id);

        Ok(())
    }

    async fn place_order(&self, p: OrderParams<'_>) -> Result<u64> {
        let mut params: Vec<(&str, String)> = vec![
            ("symbol", p.symbol.to_string()),
            ("side", p.side.to_string()),
            ("type", p.order_type.to_string()),
            ("quantity", p.qty.to_string()),
            ("timeInForce", "GTC".to_string()),
        ];
        if let Some(price) = p.price {
            params.push(("price", price.to_string()));
        }
        if let Some(sp) = p.stop_price {
            params.push(("stopPrice", sp.to_string()));
        }
        if p.close_position {
            params.push(("closePosition", "true".to_string()));
            // closePosition=true means quantity is ignored by Binance - remove it
            params.retain(|(k, _)| *k != "quantity");
        }

        let resp = self
            .send_signed_request(reqwest::Method::POST, "/fapi/v1/order", params)
            .await
            .with_context(|| "Order placement failed".to_string())
            .map_err(|e| {
                error!("Binance order error: {}", e);
                e
            })?;

        let order: OrderResponse = resp.json().await?;
        Ok(order.order_id)
    }
}

struct OrderParams<'a> {
    symbol: &'a str,
    side: &'a str,
    order_type: &'a str,
    qty: &'a str,
    price: Option<&'a str>,
    stop_price: Option<&'a str>,
    close_position: bool,
}

#[derive(Deserialize)]
struct BalanceEntry {
    pub asset: String,
    #[serde(rename = "availableBalance", deserialize_with = "de_decimal")]
    pub available_balance: Decimal,
}

#[derive(Deserialize)]
struct OrderResponse {
    #[serde(rename = "orderId")]
    pub order_id: u64,
}

#[derive(Deserialize)]
struct ServerTimeResponse {
    #[serde(rename = "serverTime")]
    pub server_time: u64,
}

fn de_decimal<'de, D>(d: D) -> std::result::Result<Decimal, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Decimal::from_str(&s).map_err(serde::de::Error::custom)
}
