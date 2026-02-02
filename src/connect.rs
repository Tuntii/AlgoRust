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
        if !resp.status().is_success() {
             anyhow::bail!("Binance API error: {}", resp.status());
        }
        let json: Vec<Vec<serde_json::Value>> = resp.json().await?;

        self.parse_candles(json)
    }

    pub async fn fetch_historical_candles(&self, symbol: &str, interval: &str, days: i64) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let limit = 1000;
        let now = Utc::now();
        let start_time = now - chrono::Duration::days(days);
        let mut current_start = start_time.timestamp_millis();
        let end_ts = now.timestamp_millis();

        tracing::info!("Fetching historical data for {} {} ({} days)...", symbol, interval, days);

        // Max requests safety break
        let max_requests = 100;
        let mut request_count = 0;

        loop {
            if current_start >= end_ts {
                break;
            }
            if request_count >= max_requests {
                tracing::warn!("Reached max request limit for historical data");
                break;
            }

            let url = format!("{}/api/v3/klines", self.base_url);
            let params = [
                ("symbol", symbol.to_string()),
                ("interval", interval.to_string()),
                ("limit", limit.to_string()),
                ("startTime", current_start.to_string()),
            ];

            // Add small delay to avoid rate limits
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let resp = self.client.get(&url).query(&params).send().await?;
            if !resp.status().is_success() {
                 tracing::error!("Binance API error fetching history: {}", resp.status());
                 // Try to continue with what we have
                 break;
            }
            
            let json: Vec<Vec<serde_json::Value>> = resp.json().await?;
            if json.is_empty() {
                break;
            }

            let candles = self.parse_candles(json)?;
            if candles.is_empty() {
                break;
            }
            
            let last_candle_time = candles.last().unwrap().open_time.timestamp_millis();
            
            // If we didn't advance, break to avoid infinite loop
             if last_candle_time <= current_start {
                break;
            }

            // Update start time for next batch (last close time + 1ms roughly, or just use next open time)
            // Ideally we use close_time + 1
            // But parse_candles returns open_time. 
            // The candle duration depends on interval.
            // Safest is to take the last candle's open time + interval duration?
            // Or just last candle open time + 1ms ? No, that would refetch the same candle roughly.
            // Binance returns candles starting FROM startTime.
            // So we should set next startTime to last_candle_close_time + 1.
            // parse_candles DOES NOT parse close_time currently (it returns None).
            // I need to update parse_candles to parse close_time or calculate it.
            // Actually, parse_candles in previous code (and my new one below) skips close_time.
            // Let's check `parse_candles`.
            
            all_candles.extend(candles);
            
            // Hack: Since I don't have close_time easily available in Candle struct (it is None),
            // I will use last open_time + interval.
            // Wait, Candle struct HAS close_time but it is Option<DateTime<Utc>>.
            // In the previous fetch_candles, it was set to None.
            // Let's Fix parse_candles to set close_time.
            
            // For now, let's assume valid candles are returned.
            // Next start time = last_candle_open_time + 1ms (Binance includes the candle covering startTime).
            // If I request startTime=T, and there is a candle at T, it returns it.
            // So next request should be T_last + 1ms.
            current_start = last_candle_time + 1; 

            request_count += 1;
        }

        tracing::info!("Fetched total {} candles for {} {}", all_candles.len(), symbol, interval);
        Ok(all_candles)
    }

    fn parse_candles(&self, json: Vec<Vec<serde_json::Value>>) -> Result<Vec<Candle>> {
        let mut candles = Vec::new();
        for row in json {
            if row.len() < 7 { continue; } // Need close time at index 6
            
            let open_time_ms = row[0].as_i64().ok_or(anyhow::anyhow!("Invalid timestamp"))?;
            let open_time = DateTime::<Utc>::from_utc(
                chrono::NaiveDateTime::from_timestamp_millis(open_time_ms).unwrap(), 
                Utc
            );

            let open: Decimal = row[1].as_str().unwrap_or("0").parse()?;
            let high: Decimal = row[2].as_str().unwrap_or("0").parse()?;
            let low: Decimal = row[3].as_str().unwrap_or("0").parse()?;
            let close: Decimal = row[4].as_str().unwrap_or("0").parse()?;
            let volume: Decimal = row[5].as_str().unwrap_or("0").parse()?;
            
            let close_time_ms = row[6].as_i64().unwrap_or(0);
             let close_time = if close_time_ms > 0 {
                Some(DateTime::<Utc>::from_utc(
                    chrono::NaiveDateTime::from_timestamp_millis(close_time_ms).unwrap(), 
                    Utc
                ))
            } else {
                None
            };

            candles.push(Candle {
                open_time,
                open,
                high,
                low,
                close,
                volume,
                close_time, 
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
