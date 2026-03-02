// ============================================================================
// BINANCE FUTURES TRADER
// Real-money order execution via Binance USDM Futures REST API
// Endpoints: POST /fapi/v1/order, POST /fapi/v1/algoOrder, GET /fapi/v2/balance
// Auth: HMAC-SHA256 signature on every signed request
// ============================================================================

use crate::state::SymbolContext;
use crate::types::{SignalType, TradeSignal};
use anyhow::{Context, Result};

/// Result of execute_signal — tells caller whether a flip happened
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalExecResult {
    /// New position opened (no prior position)
    Executed,
    /// Prior opposite position closed + new one opened
    Flipped,
    /// Same direction already open — skipped
    Skipped,
}
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::Sha256;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};
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
    /// Fixed risk amount per trade in USDT (e.g. 5.0)
    pub risk_amount: Decimal,
    /// Leverage applied to every position (e.g. 1 = 1×)
    pub leverage: u32,
    /// "sl_tp" = place SL+TP orders, "indicator_flip" = only SL, exit via opposite signal
    pub live_exit_mode: String,
    /// Aynı anda açılabilen maksimum pozisyon sayısı.
    /// Margin cap hesabında bakiye bu sayıya bölünür; böylece
    /// farklı semboller için yeterli teminat rezerve edilir.
    pub max_positions: u32,
    /// Trailing stop aktif mi? (indicator_flip modunda kâr koruma)
    pub trailing_stop_enabled: bool,
    /// Trailing stop callback oranı (% — 0.1–5.0)
    pub trailing_callback_rate: Decimal,
    /// TP mesafesinin yüzde kaçında trailing stop aktif olacak (0.0–1.0)
    pub trailing_activation_pct: Decimal,
}

impl BinanceFuturesTrader {
    /// Load credentials from .env:
    ///   BINANCE_API_KEY, BINANCE_API_SECRET
    ///   BINANCE_TESTNET=true  (optional, uses testnet if set)
    pub fn new(
        risk_amount: Decimal,
        leverage: u32,
        live_exit_mode: String,
        max_positions: u32,
        trailing_stop_enabled: bool,
        trailing_callback_rate: f64,
        trailing_activation_pct: f64,
    ) -> Result<Self> {
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
            risk_amount,
            leverage,
            live_exit_mode,
            max_positions: max_positions.max(1),
            trailing_stop_enabled,
            trailing_callback_rate: Decimal::from_f64(trailing_callback_rate)
                .unwrap_or(Decimal::new(4, 1)), // 0.4%
            trailing_activation_pct: Decimal::from_f64(trailing_activation_pct)
                .unwrap_or(Decimal::new(35, 2)), // 0.35
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

    /// Risk per trade = Fixed Amount (1R)
    /// Qty = Risk Amount / |Entry - SL|
    /// Capped so that notional (qty × entry) never exceeds available margin
    /// Risk per trade = Fixed Amount (1R)
    /// Qty = Risk Amount / |Entry - SL|
    /// Capped so that notional (qty × entry) never exceeds available margin
    pub async fn position_qty(
        &self,
        entry: Decimal,
        sl: Decimal,
        qty_precision: u32,
        min_notional: Decimal,
    ) -> Result<Decimal> {
        let balance = self.usdt_balance().await?;

        // 1. Calculate SL distance per unit
        let sl_distance = (entry - sl).abs();
        if sl_distance.is_zero() {
            warn!("SL distance is zero, using minimum qty 0.001");
            return Ok(Decimal::new(1, qty_precision));
        }

        // 2. Calculate quantity based on fixed dollar risk
        let risk_qty = self.risk_amount / sl_distance;

        // 3. Margin cap: at Nx leverage, max notional = (balance / max_positions) * N
        // Balance'ı max_positions'a bölerek her sembol için eşit teminat rezerve edilir.
        // Örn: balance=$90, max_positions=2, leverage=2 → per-slot notional=$90
        // Leave a 2% cushion for fees & funding
        let leverage_dec = Decimal::from(self.leverage);
        let slots = Decimal::from(self.max_positions.max(1));
        let per_slot_balance = balance / slots;
        let max_notional = per_slot_balance * leverage_dec * Decimal::new(98, 2); // × 0.98
        let max_qty_by_margin = if entry.is_zero() {
            risk_qty
        } else {
            max_notional / entry
        };

        // 4. Take the smaller of the two (Risk-based vs Wallet-based)
        let qty = risk_qty.min(max_qty_by_margin);

        // Round to precision
        let qty = qty.round_dp(qty_precision);

        info!(
            "Size calc: balance=${} slots={} per_slot=${} leverage={}x risk_amount=${} sl_dist={} \
             risk_qty={} margin_cap_qty={} -> final_qty={} (prec={})",
            balance,
            self.max_positions,
            per_slot_balance,
            self.leverage,
            self.risk_amount,
            sl_distance,
            risk_qty,
            max_qty_by_margin,
            qty,
            qty_precision
        );

        // Ensure minimum based on min_notional
        let notional = qty * entry;
        if notional < min_notional {
            // If the calculated qty is less than min_notional, it will fail.
            // But if we just bump it to min_notional, we might exceed margin!
            // Let's see what qty we'd need for min_notional
            let mut bumped_qty = if entry.is_zero() {
                Decimal::ZERO
            } else {
                (min_notional / entry)
                    // Bump up slightly to account for funding/fees edge cases
                    * Decimal::new(101, 2)
            };
            bumped_qty = bumped_qty.round_dp(qty_precision);

            // Is bumped_qty allowed by margin?
            if bumped_qty <= max_qty_by_margin {
                warn!(
                    "Calculated notional ({} USDT) < minNotional ({} USDT). Bumping qty to {}.",
                    notional, min_notional, bumped_qty
                );
                return Ok(bumped_qty);
            } else {
                warn!(
                    "Calculated notional ({} USDT) < min_notional ({} USDT), and margin cap prevents bumping. Skipping trade.",
                    notional, min_notional
                );
                return Ok(Decimal::ZERO);
            }
        }

        // Ensure minimum > 0 if notional is okay (fallback)
        if qty.is_zero() {
            let min_qty = Decimal::new(1, qty_precision);
            return Ok(min_qty);
        }

        Ok(qty)
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
    pub async fn execute_signal(&self, signal: &TradeSignal, ctx: &SymbolContext) -> Result<SignalExecResult> {
        // Fetch precisions dynamically
        let (price_prec, qty_prec, min_notional) = self
            .get_precisions(&signal.symbol)
            .await
            .unwrap_or((2, 3, Decimal::from(100)));

        // Guard live exposure per symbol:
        // - same-direction signal => skip
        // - opposite-direction signal => cancel old exits, close old position, then flip
        let desired_dir = match signal.signal {
            SignalType::LONG => 1,
            SignalType::SHORT => -1,
        };

        let net_pos_qty = self
            .net_position_qty(&signal.symbol)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    "Could not read current position for {} (continuing): {}",
                    signal.symbol, e
                );
                Decimal::ZERO
            });

        if !net_pos_qty.is_zero() {
            let current_dir = if net_pos_qty > Decimal::ZERO { 1 } else { -1 };
            if current_dir == desired_dir {
                info!(
                    "Skipping {} {} signal: existing {} position still open (qty={}).",
                    signal.signal,
                    signal.symbol,
                    if current_dir == 1 { "LONG" } else { "SHORT" },
                    net_pos_qty.abs()
                );
                return Ok(SignalExecResult::Skipped);
            }

            info!(
                "Opposite signal on {} with open qty={}; closing existing position first.",
                signal.symbol, net_pos_qty
            );
            self.cancel_symbol_open_orders(&signal.symbol).await;
            self.close_net_position_market(&signal.symbol, net_pos_qty, qty_prec)
                .await?;

            let is_flat = self
                .wait_until_flat(&signal.symbol, 5, 200)
                .await
                .unwrap_or(false);
            if !is_flat {
                warn!(
                    "Position on {} not flat after close attempt. Skipping new entry for safety.",
                    signal.symbol
                );
                return Ok(SignalExecResult::Skipped);
            }
        } else {
            // If no position exists, clear stale exit orders that can block future flips.
            self.cancel_symbol_open_orders(&signal.symbol).await;
        }

        // Enforce configured leverage before every new entry
        self.set_leverage(&signal.symbol, self.leverage).await?;
        let entry = signal.price;

        let (sl, tp) = crate::alpaca::calculate_sl_tp_pub(signal, ctx, entry);

        let qty = self.position_qty(entry, sl, qty_prec, min_notional).await?;

        if qty.is_zero() {
            warn!("Position quantity is zero (or below minNotional). Trade skipped.");
            return Ok(SignalExecResult::Skipped);
        }

        // Track whether this was a flip (opposite close + new open)
        let was_flip = !net_pos_qty.is_zero();

        let (side, sl_side, tp_side) = match signal.signal {
            SignalType::LONG => ("BUY", "SELL", "SELL"),
            SignalType::SHORT => ("SELL", "BUY", "BUY"),
        };

        // Format with correct precision
        let sl_str = format!("{:.1$}", sl, price_prec as usize);
        let qty_str = format!("{:.1$}", qty, qty_prec as usize);

        let use_tp = self.live_exit_mode != "indicator_flip";

        if use_tp {
            let tp_str = format!("{:.1$}", tp, price_prec as usize);
            info!(
                "Binance Futures signal: {} {} qty={} entry={} SL={} TP={}",
                side, signal.symbol, qty_str, entry, sl_str, tp_str
            );
        } else {
            if self.trailing_stop_enabled {
                let tp_distance = (tp - entry).abs();
                let act_offset = tp_distance * self.trailing_activation_pct;
                let act_price = match signal.signal {
                    SignalType::LONG => entry + act_offset,
                    SignalType::SHORT => entry - act_offset,
                };
                info!(
                    "Binance Futures signal: {} {} qty={} entry={} SL={} (trailing: act={} cb={}%)",
                    side, signal.symbol, qty_str, entry, sl_str,
                    format!("{:.1$}", act_price, price_prec as usize),
                    self.trailing_callback_rate
                );
            } else {
                info!(
                    "Binance Futures signal: {} {} qty={} entry={} SL={} (indicator_flip: no TP)",
                    side, signal.symbol, qty_str, entry, sl_str
                );
            }
        }

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
                reduce_only: false,
                callback_rate: None,
                activation_price: None,
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
                reduce_only: false,
                callback_rate: None,
                activation_price: None,
            })
            .await?;
        info!("Stop-loss placed @ {} - id={}", sl_str, sl_id);

        // 3. Take-profit (only in sl_tp mode; indicator_flip exits via opposite signal)
        if use_tp {
            let tp_str = format!("{:.1$}", tp, price_prec as usize);
            let tp_id = self
                .place_order(OrderParams {
                    symbol: &signal.symbol,
                    side: tp_side,
                    order_type: "TAKE_PROFIT_MARKET",
                    qty: &qty_str,
                    price: None,
                    stop_price: Some(&tp_str),
                    close_position: true,
                    reduce_only: false,
                    callback_rate: None,
                    activation_price: None,
                })
                .await?;
            info!("Take-profit placed @ {} - id={}", tp_str, tp_id);
        }

        // 4. Trailing stop — indicator_flip modunda kâr koruması
        // Fiyat TP mesafesinin %activation'una ulaşınca trailing stop aktif olur.
        // Böylece KAMA flip'ini beklemeden kâr otomatik kilitlenir.
        if !use_tp && self.trailing_stop_enabled {
            // Activation price = entry + (tp - entry) * activation_pct
            let tp_distance = (tp - entry).abs();
            let activation_offset = tp_distance * self.trailing_activation_pct;
            let activation_price = match signal.signal {
                SignalType::LONG => entry + activation_offset,
                SignalType::SHORT => entry - activation_offset,
            };
            let activation_str = format!("{:.1$}", activation_price, price_prec as usize);
            let callback_str = format!("{}", self.trailing_callback_rate);

            info!(
                "Trailing stop config: TP_dist={} activation_offset={} activation_price={} callback={}%",
                tp_distance, activation_offset, activation_str, callback_str
            );

            match self
                .place_order(OrderParams {
                    symbol: &signal.symbol,
                    side: sl_side,
                    order_type: "TRAILING_STOP_MARKET",
                    qty: &qty_str,
                    price: None,
                    stop_price: None,
                    close_position: false,
                    reduce_only: false,
                    callback_rate: Some(&callback_str),
                    activation_price: Some(&activation_str),
                })
                .await
            {
                Ok(trail_id) => {
                    info!(
                        "Trailing stop placed: activation={} callback={}% - id={}",
                        activation_str, callback_str, trail_id
                    );
                }
                Err(e) => {
                    // Trailing stop hatası pozisyonu iptal etmemeli, sadece uyarı ver
                    warn!(
                        "⚠️ Trailing stop order failed (position still protected by SL): {}",
                        e
                    );
                }
            }
        }

        Ok(if was_flip {
            SignalExecResult::Flipped
        } else {
            SignalExecResult::Executed
        })
    }

    pub async fn place_test_order(&self, symbol: &str, risk_usdt: Decimal) -> Result<u64> {
        // 1. Get price
        let price_url = format!("{}/fapi/v1/ticker/price?symbol={}", self.base_url, symbol);
        let resp = self.client.get(&price_url).send().await?;
        let price_json: serde_json::Value = resp.json().await?;
        let price_str = price_json["price"].as_str().context("No price found")?;
        let price = Decimal::from_str(price_str)?;

        info!("Test Order: {} price = {}", symbol, price);

        // 2. Calculate quantity: risk_usdt / price
        // Fetch precision dynamically
        let (_, qty_prec, _) =
            self.get_precisions(symbol)
                .await
                .unwrap_or((2, 3, Decimal::from(100)));

        let qty = (risk_usdt / price).round_dp(qty_prec);
        let qty_str = format!("{:.1$}", qty, qty_prec as usize);

        info!(
            "Test Order: Placing MARKET BUY for {} {} (${}) (prec={})",
            qty_str, symbol, risk_usdt, qty_prec
        );

        self.place_order(OrderParams {
            symbol,
            side: "BUY",
            order_type: "MARKET",
            qty: &qty_str,
            price: None,
            stop_price: None,
            close_position: false,
            reduce_only: false,
            callback_rate: None,
            activation_price: None,
        })
        .await
    }

    pub async fn get_precisions(&self, symbol: &str) -> Result<(u32, u32, Decimal)> {
        // Try to fetch info for just this symbol
        let url = format!("{}/fapi/v1/exchangeInfo?symbol={}", self.base_url, symbol);
        let resp = self.client.get(&url).send().await?;
        let info: serde_json::Value = resp.json().await?;

        // If query param works, "symbols" has 1 item.
        if let Some(symbols) = info["symbols"].as_array() {
            for s in symbols {
                if s["symbol"] == symbol {
                    let q_prec = s["quantityPrecision"].as_u64().unwrap_or(3) as u32;
                    let p_prec = s["pricePrecision"].as_u64().unwrap_or(2) as u32;

                    let mut min_notional = Decimal::from(5); // default for most
                    if let Some(filters) = s["filters"].as_array() {
                        for f in filters {
                            if f["filterType"] == "MIN_NOTIONAL" {
                                if let Some(n) = f["notional"].as_str() {
                                    if let Ok(dec) = Decimal::from_str(n) {
                                        min_notional = dec;
                                    }
                                }
                            }
                        }
                    }
                    return Ok((p_prec, q_prec, min_notional));
                }
            }
        }
        Ok((2, 3, Decimal::from(5)))
    }

    async fn net_position_qty(&self, symbol: &str) -> Result<Decimal> {
        let params = vec![("symbol", symbol.to_string())];

        let resp = match self
            .send_signed_request(
                reqwest::Method::GET,
                "/fapi/v3/positionRisk",
                params.clone(),
            )
            .await
        {
            Ok(r) => r,
            Err(v3_err) => {
                warn!(
                    "positionRisk v3 failed for {} ({}), falling back to v2",
                    symbol, v3_err
                );
                self.send_signed_request(reqwest::Method::GET, "/fapi/v2/positionRisk", params)
                    .await?
            }
        };

        let entries: Vec<PositionRiskEntry> = resp.json().await?;

        let mut net_qty = Decimal::ZERO;
        let mut long_qty = Decimal::ZERO;
        let mut short_qty = Decimal::ZERO;

        for p in entries.into_iter().filter(|p| p.symbol == symbol) {
            match p.position_side.as_str() {
                "BOTH" => net_qty += p.position_amt,
                "LONG" => long_qty += p.position_amt.abs(),
                "SHORT" => short_qty += p.position_amt.abs(),
                _ => net_qty += p.position_amt,
            }
        }

        if net_qty.is_zero() && (!long_qty.is_zero() || !short_qty.is_zero()) {
            if !long_qty.is_zero() && !short_qty.is_zero() {
                warn!(
                    "Hedge-style positions detected on {} (LONG={} SHORT={}); using net.",
                    symbol, long_qty, short_qty
                );
            }
            net_qty = long_qty - short_qty;
        }

        Ok(net_qty)
    }

    async fn cancel_symbol_open_orders(&self, symbol: &str) {
        let params = vec![("symbol", symbol.to_string())];

        match self
            .send_signed_request(
                reqwest::Method::DELETE,
                "/fapi/v1/allOpenOrders",
                params.clone(),
            )
            .await
        {
            Ok(_) => info!("Canceled open regular orders for {}", symbol),
            Err(e) => warn!("Could not cancel regular orders for {}: {}", symbol, e),
        }

        match self
            .send_signed_request(reqwest::Method::DELETE, "/fapi/v1/algoOpenOrders", params)
            .await
        {
            Ok(_) => info!("Canceled open algo orders for {}", symbol),
            Err(e) => warn!("Could not cancel algo orders for {}: {}", symbol, e),
        }
    }

    async fn close_net_position_market(
        &self,
        symbol: &str,
        net_qty: Decimal,
        qty_precision: u32,
    ) -> Result<()> {
        if net_qty.is_zero() {
            return Ok(());
        }

        let close_side = if net_qty > Decimal::ZERO {
            "SELL"
        } else {
            "BUY"
        };
        let close_qty = net_qty.abs().round_dp(qty_precision);
        if close_qty.is_zero() {
            warn!(
                "Computed close qty rounded to zero for {} (net_qty={})",
                symbol, net_qty
            );
            return Ok(());
        }
        let close_qty_str = format!("{:.1$}", close_qty, qty_precision as usize);

        let close_id = self
            .place_order(OrderParams {
                symbol,
                side: close_side,
                order_type: "MARKET",
                qty: &close_qty_str,
                price: None,
                stop_price: None,
                close_position: false,
                reduce_only: true,
                callback_rate: None,
                activation_price: None,
            })
            .await?;

        info!(
            "Position close sent: {} {} qty={} orderId={}",
            close_side, symbol, close_qty_str, close_id
        );
        Ok(())
    }

    async fn wait_until_flat(&self, symbol: &str, retries: usize, delay_ms: u64) -> Result<bool> {
        for _ in 0..retries {
            let qty = self.net_position_qty(symbol).await?;
            if qty.is_zero() {
                return Ok(true);
            }
            sleep(Duration::from_millis(delay_ms)).await;
        }

        Ok(false)
    }

    async fn place_order(&self, p: OrderParams<'_>) -> Result<u64> {
        // Since Dec 2025, conditional futures orders must go through /fapi/v1/algoOrder.
        let is_algo_order = is_algo_order_type(p.order_type);
        let needs_tif = if is_algo_order {
            matches!(p.order_type, "STOP" | "TAKE_PROFIT")
        } else {
            matches!(p.order_type, "LIMIT")
        };

        let mut params: Vec<(&str, String)> = vec![
            ("symbol", p.symbol.to_string()),
            ("side", p.side.to_string()),
            ("type", p.order_type.to_string()),
            ("quantity", p.qty.to_string()),
        ];
        if is_algo_order {
            params.push(("algoType", "CONDITIONAL".to_string()));
        }
        if needs_tif {
            params.push(("timeInForce", "GTC".to_string()));
        }
        if let Some(price) = p.price {
            params.push(("price", price.to_string()));
        }
        if let Some(sp) = p.stop_price {
            let trigger_key = if is_algo_order {
                "triggerPrice"
            } else {
                "stopPrice"
            };
            params.push((trigger_key, sp.to_string()));
        }
        if let Some(cb) = p.callback_rate {
            params.push(("callbackRate", cb.to_string()));
        }
        if let Some(ap) = p.activation_price {
            params.push(("activationPrice", ap.to_string()));
        }
        if p.close_position {
            params.push(("closePosition", "true".to_string()));
            // closePosition=true means Binance ignores quantity - remove it to avoid -1102
            params.retain(|(k, _)| *k != "quantity");
        }
        if p.reduce_only && !is_algo_order {
            params.push(("reduceOnly", "true".to_string()));
        }

        let endpoint = if is_algo_order {
            "/fapi/v1/algoOrder"
        } else {
            "/fapi/v1/order"
        };

        let resp = self
            .send_signed_request(reqwest::Method::POST, endpoint, params)
            .await
            .map_err(|e| {
                // The real Binance error body is embedded in `e` - log it fully
                error!(
                    "Binance order error ({} {} via {}): {:#}",
                    p.order_type, p.symbol, endpoint, e
                );
                e
            })?;

        let payload: serde_json::Value = resp.json().await?;
        let order_id = parse_u64_field(&payload, "orderId");
        let algo_id = parse_u64_field(&payload, "algoId");

        order_id.or(algo_id).with_context(|| {
            format!(
                "order response missing orderId/algoId (endpoint={} payload={})",
                endpoint, payload
            )
        })
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
    reduce_only: bool,
    /// TRAILING_STOP_MARKET: callback rate in % (e.g. "0.4" = 0.4%)
    callback_rate: Option<&'a str>,
    /// TRAILING_STOP_MARKET: activation price (trailing starts when price reaches this)
    activation_price: Option<&'a str>,
}

#[derive(Deserialize)]
struct BalanceEntry {
    pub asset: String,
    #[serde(rename = "availableBalance", deserialize_with = "de_decimal")]
    pub available_balance: Decimal,
}

#[derive(Deserialize)]
struct ServerTimeResponse {
    #[serde(rename = "serverTime")]
    pub server_time: u64,
}

fn is_algo_order_type(order_type: &str) -> bool {
    matches!(
        order_type,
        "STOP_MARKET" | "TAKE_PROFIT_MARKET" | "STOP" | "TAKE_PROFIT" | "TRAILING_STOP_MARKET"
    )
}

fn parse_u64_field(payload: &serde_json::Value, key: &str) -> Option<u64> {
    payload.get(key).and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
    })
}

#[derive(Deserialize)]
struct PositionRiskEntry {
    pub symbol: String,
    #[serde(rename = "positionSide")]
    pub position_side: String,
    #[serde(rename = "positionAmt", deserialize_with = "de_decimal")]
    pub position_amt: Decimal,
}

fn de_decimal<'de, D>(d: D) -> std::result::Result<Decimal, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Decimal::from_str(&s).map_err(serde::de::Error::custom)
}
