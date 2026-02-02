use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub open_time: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    #[serde(default)] 
    pub close_time: Option<DateTime<Utc>>,
}

impl Candle {
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    LONG,
    SHORT,
}

#[derive(Debug, Clone, Serialize)]
pub struct TradeSignal {
    pub symbol: String,
    pub timeframe: String,
    pub signal: SignalType,
    pub price: Decimal,
    pub confidence: u8, // 0-100
    pub timestamp: DateTime<Utc>,
    pub reasons: Vec<String>,
}

impl fmt::Display for TradeSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, 
            "SIGNAL: {} [{}] {:?} @ {} (Conf: {}%)", 
            self.symbol, self.timeframe, self.signal, self.price, self.confidence
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendState {
    Bullish,
    Bearish,
    Neutral,
}

#[derive(Debug, Clone)]
pub struct MarketStructure {
    pub last_pivot_high: Option<Decimal>,
    pub last_pivot_low: Option<Decimal>,
    pub trend: TrendState,
    pub bos_confirmed: bool,
}

impl Default for MarketStructure {
    fn default() -> Self {
        Self {
            last_pivot_high: None,
            last_pivot_low: None,
            trend: TrendState::Neutral,
            bos_confirmed: false,
        }
    }
}

// WebSocket Message Types
#[derive(Debug, Deserialize)]
pub struct WsStreamMessage {
    pub stream: String,
    pub data: WsKlineEvent,
}

#[derive(Debug, Deserialize)]
pub struct WsKlineEvent {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "k")]
    pub kline: WsKline,
}

#[derive(Debug, Deserialize)]
pub struct WsKline {
    #[serde(rename = "t")]
    pub start_time: i64,
    #[serde(rename = "T")]
    pub end_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "i")]
    pub interval: String,
    #[serde(rename = "o")]
    pub open: String,
    #[serde(rename = "c")]
    pub close: String,
    #[serde(rename = "h")]
    pub high: String,
    #[serde(rename = "l")]
    pub low: String,
    #[serde(rename = "v")]
    pub volume: String,
    #[serde(rename = "x")]
    pub is_closed: bool,
}

impl WsKline {
    pub fn to_candle(&self) -> anyhow::Result<Candle> {
        Ok(Candle {
            open_time: DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(self.start_time).unwrap(),
                Utc
            ),
            open: Decimal::from_str_exact(&self.open)?,
            high: Decimal::from_str_exact(&self.high)?,
            low: Decimal::from_str_exact(&self.low)?,
            close: Decimal::from_str_exact(&self.close)?,
            volume: Decimal::from_str_exact(&self.volume)?,
            close_time: Some(DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(self.end_time).unwrap(),
                Utc
            )),
        })
    }
}

