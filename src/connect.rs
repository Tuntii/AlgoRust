use crate::types::Candle;
use anyhow::Result;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;
use serde::Deserialize;

pub struct BinanceClient {
    base_url: String,
    client: reqwest::Client,
}

impl BinanceClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.binance.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn fetch_candles(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let url = format!("{}/api/v3/klines", self.base_url);
        let params = [
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.to_string()),
        ];

        let resp = self.client.get(&url).query(&params).send().await?;
        let json: Vec<Vec<serde_json::Value>> = resp.json().await?;

        let mut candles = Vec::new();
        for row in json {
            // Binance response: [open_time, open, high, low, close, volume, close_time, ...]
            if row.len() < 6 { continue; }
            
            let open_time_ms = row[0].as_i64().ok_or(anyhow::anyhow!("Invalid timestamp"))?;
            let open_time = DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(open_time_ms).unwrap(), 
                Utc
            );

            let open: Decimal = row[1].as_str().unwrap().parse()?;
            let high: Decimal = row[2].as_str().unwrap().parse()?;
            let low: Decimal = row[3].as_str().unwrap().parse()?;
            let close: Decimal = row[4].as_str().unwrap().parse()?;
            let volume: Decimal = row[5].as_str().unwrap().parse()?;

            candles.push(Candle {
                open_time,
                open,
                high,
                low,
                close,
                volume,
                close_time: None,
            });
        }
        
        Ok(candles)
    }
}

// WebSocket implementation
use futures_util::{Stream, SinkExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{WebSocketStream, MaybeTlsStream};
use tracing::info;

pub async fn connect_stream(symbols: &[String], intervals: &[String]) -> Result<impl Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>> {
    let mut streams = Vec::new();
    for s in symbols {
        for i in intervals {
            let stream_name = format!("{}@kline_{}", s.to_lowercase(), i);
            streams.push(stream_name);
        }
    }
    let stream_query = streams.join("/");
    // Binance stream URL: wss://stream.binance.com:9443/stream?streams=<streamName1>/<streamName2>...
    let url_str = format!("wss://stream.binance.com:9443/stream?streams={}", stream_query);
    let url = Url::parse(&url_str)?;

    info!("Connecting to WebSocket: {}", url);
    let (ws_stream, _) = connect_async(url).await?;
    
    Ok(ws_stream)
}
