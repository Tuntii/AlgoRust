use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::fmt;

// =============================================================================
// MULTI-POSITION TRADING TYPES
// =============================================================================

/// Unique identifier for a trading context (BOS, liquidity sweep, etc.)
/// This prevents entering the same setup multiple times
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextId {
    /// Type of context: "bos", "liquidity_sweep", "pivot"
    pub context_type: String,
    /// Unique identifier (e.g., pivot timestamp, BOS index)
    pub identifier: String,
    /// The candle index when this context was created
    pub created_at_candle: usize,
}

impl ContextId {
    pub fn new(context_type: &str, identifier: &str, candle_idx: usize) -> Self {
        Self {
            context_type: context_type.to_string(),
            identifier: identifier.to_string(),
            created_at_candle: candle_idx,
        }
    }
    
    /// Create context ID from BOS event
    pub fn from_bos(candle_idx: usize, is_bullish: bool) -> Self {
        Self::new(
            "bos",
            &format!("{}_{}", if is_bullish { "bull" } else { "bear" }, candle_idx),
            candle_idx,
        )
    }
    
    /// Create context ID from liquidity sweep
    pub fn from_liquidity_sweep(candle_idx: usize, level: Decimal) -> Self {
        Self::new(
            "liquidity_sweep",
            &format!("{}_{}", level, candle_idx),
            candle_idx,
        )
    }
    
    /// Create context ID from pivot confirmation
    pub fn from_pivot(candle_idx: usize, is_high: bool) -> Self {
        Self::new(
            "pivot",
            &format!("{}_{}", if is_high { "high" } else { "low" }, candle_idx),
            candle_idx,
        )
    }
}

impl fmt::Display for ContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.context_type, self.identifier)
    }
}

/// Active trade in the position pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTrade {
    /// Unique trade ID
    pub trade_id: String,
    /// The signal that triggered this trade
    pub signal: TradeSignal,
    /// Entry price
    pub entry_price: Decimal,
    /// Stop loss price
    pub sl_price: Decimal,
    /// Take profit price
    pub tp_price: Decimal,
    /// Direction of the trade
    pub direction: SignalType,
    /// Context ID that generated this trade (for uniqueness check)
    pub context_id: ContextId,
    /// Candle index when trade was opened
    pub opened_at_candle: usize,
    /// Exit price (if closed)
    pub exit_price: Option<Decimal>,
    /// PnL in R (if closed)
    pub pnl_r: Option<Decimal>,
    /// Outcome: "WIN", "LOSS", "BE" (if closed)
    pub outcome: Option<String>,
    /// Candle index when trade was closed
    pub exit_candle_idx: Option<usize>,
    /// Duration in candles
    pub duration_candles: Option<u32>,
    /// Confidence adjustment based on concurrent trades
    pub adjusted_confidence: u8,
}

impl ActiveTrade {
    pub fn new(
        signal: TradeSignal,
        entry_price: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
        context_id: ContextId,
        opened_at_candle: usize,
    ) -> Self {
        use uuid::Uuid;
        
        let direction = signal.signal.clone();
        let confidence = signal.confidence;
        
        Self {
            trade_id: Uuid::new_v4().to_string(),
            signal,
            entry_price,
            sl_price,
            tp_price,
            direction,
            context_id,
            opened_at_candle,
            exit_price: None,
            pnl_r: None,
            outcome: None,
            exit_candle_idx: None,
            duration_candles: None,
            adjusted_confidence: confidence,
        }
    }
    
    /// Check if trade is still open
    pub fn is_open(&self) -> bool {
        self.outcome.is_none()
    }
    
    /// Close the trade with given outcome
    pub fn close(&mut self, exit_price: Decimal, pnl_r: Decimal, outcome: &str, exit_candle_idx: usize) {
        self.exit_price = Some(exit_price);
        self.pnl_r = Some(pnl_r);
        self.outcome = Some(outcome.to_string());
        self.exit_candle_idx = Some(exit_candle_idx);
        self.duration_candles = Some((exit_candle_idx - self.opened_at_candle) as u32);
    }
}

/// Multi-position pool configuration
#[derive(Debug, Clone)]
pub struct PositionPoolConfig {
    /// Maximum active trades per symbol/timeframe
    pub max_active_trades_per_symbol_tf: usize,
    /// Allow same direction trades (LONG + LONG)
    pub allow_same_direction: bool,
    /// Allow opposite direction trades (LONG + SHORT) - ALWAYS FALSE per strategy
    pub allow_hedge: bool,
    /// Confidence reduction per additional trade
    pub confidence_reduction_per_trade: f64,
}

impl Default for PositionPoolConfig {
    fn default() -> Self {
        Self {
            max_active_trades_per_symbol_tf: 2,  // Max 2 trades per symbol/TF
            allow_same_direction: true,           // LONG + LONG allowed (continuation)
            allow_hedge: false,                   // LONG + SHORT NOT allowed
            confidence_reduction_per_trade: 0.4,  // 40% reduction per additional trade
        }
    }
}

/// Position pool manager for multi-position trading
#[derive(Debug, Clone, Default)]
pub struct PositionPool {
    /// All trades (open and closed)
    pub trades: Vec<ActiveTrade>,
    /// Configuration
    pub config: PositionPoolConfig,
}

impl PositionPool {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_config(config: PositionPoolConfig) -> Self {
        Self {
            trades: Vec::new(),
            config,
        }
    }
    
    /// Get all active (open) trades for a symbol/timeframe
    pub fn active_trades(&self, symbol: &str, timeframe: &str) -> Vec<&ActiveTrade> {
        self.trades.iter()
            .filter(|t| t.is_open() && t.signal.symbol == symbol && t.signal.timeframe == timeframe)
            .collect()
    }
    
    /// Get count of active trades for symbol/timeframe
    pub fn active_count(&self, symbol: &str, timeframe: &str) -> usize {
        self.active_trades(symbol, timeframe).len()
    }
    
    /// Check if we can open a new trade (all guards)
    pub fn can_open_trade(
        &self, 
        symbol: &str, 
        timeframe: &str, 
        direction: &SignalType,
        context_id: &ContextId,
    ) -> (bool, Option<String>) {
        let active = self.active_trades(symbol, timeframe);
        
        // Guard 1: Max active trades
        if active.len() >= self.config.max_active_trades_per_symbol_tf {
            return (false, Some(format!(
                "Max active trades ({}) reached for {} {}",
                self.config.max_active_trades_per_symbol_tf, symbol, timeframe
            )));
        }
        
        // Guard 2: Context uniqueness - no duplicate context_id
        for trade in &active {
            if trade.context_id == *context_id {
                return (false, Some(format!(
                    "Context ID {} already has an active trade",
                    context_id
                )));
            }
        }
        
        // Guard 3: No hedge (opposite direction)
        if !self.config.allow_hedge {
            for trade in &active {
                if trade.direction != *direction {
                    return (false, Some(format!(
                        "Hedge not allowed: existing {:?} trade, attempted {:?}",
                        trade.direction, direction
                    )));
                }
            }
        }
        
        (true, None)
    }
    
    /// Calculate adjusted confidence based on active trades
    pub fn calculate_adjusted_confidence(&self, symbol: &str, timeframe: &str, base_confidence: u8) -> u8 {
        let active_count = self.active_count(symbol, timeframe);
        
        if active_count == 0 {
            return base_confidence;
        }
        
        // Reduce confidence by configured amount per active trade
        // With 2 trades: confidence = base * 0.6
        let multiplier = 1.0 - (active_count as f64 * self.config.confidence_reduction_per_trade);
        let adjusted = (base_confidence as f64 * multiplier.max(0.2)) as u8;
        adjusted.max(1)
    }
    
    /// Add a new trade to the pool
    pub fn add_trade(&mut self, trade: ActiveTrade) {
        self.trades.push(trade);
    }
    
    /// Get mutable reference to active trades for exit processing
    pub fn active_trades_mut(&mut self) -> Vec<&mut ActiveTrade> {
        self.trades.iter_mut()
            .filter(|t| t.is_open())
            .collect()
    }
    
    /// Get all completed trades
    pub fn completed_trades(&self) -> Vec<&ActiveTrade> {
        self.trades.iter()
            .filter(|t| !t.is_open())
            .collect()
    }
    
    /// Get max concurrent trades (for metrics)
    pub fn max_concurrent_trades(&self) -> usize {
        if self.trades.is_empty() {
            return 0;
        }
        
        // Find the maximum number of overlapping trades
        let mut max_concurrent = 0;
        
        for trade in &self.trades {
            let start = trade.opened_at_candle;
            let end = trade.exit_candle_idx.unwrap_or(usize::MAX);
            
            let concurrent = self.trades.iter()
                .filter(|t| {
                    let t_start = t.opened_at_candle;
                    let t_end = t.exit_candle_idx.unwrap_or(usize::MAX);
                    // Check overlap
                    t_start <= end && t_end >= start
                })
                .count();
            
            max_concurrent = max_concurrent.max(concurrent);
        }
        
        max_concurrent
    }
    
    /// Get average concurrent trades (for metrics)
    pub fn avg_concurrent_trades(&self) -> f64 {
        let completed = self.completed_trades();
        if completed.is_empty() {
            return 0.0;
        }
        
        let total_overlap: usize = completed.iter()
            .map(|t| {
                let start = t.opened_at_candle;
                let end = t.exit_candle_idx.unwrap_or(start);
                
                completed.iter()
                    .filter(|other| {
                        let o_start = other.opened_at_candle;
                        let o_end = other.exit_candle_idx.unwrap_or(o_start);
                        o_start <= end && o_end >= start && other.trade_id != t.trade_id
                    })
                    .count()
            })
            .sum();
        
        1.0 + (total_overlap as f64 / (completed.len() as f64 * 2.0))
    }
}

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

// T6.2 — Signal Output Contracts
pub const ENGINE_VERSION: &str = "2.0.0";

#[derive(Debug, Clone, Serialize)]
pub struct TradeSignal {
    pub signal_id: String,           // T6.2: Unique signal ID
    pub engine_version: String,      // T6.2: Engine version
    pub symbol: String,
    pub timeframe: String,
    pub signal: SignalType,
    pub price: Decimal,
    pub confidence: u8,              // 0-100 (T6.1: normalized score)
    pub confidence_tier: String,     // T6.1: "high", "medium", "low"
    pub timestamp: DateTime<Utc>,
    pub reasons: Vec<String>,
    pub regime_context: Option<RegimeContext>, // T5.2: Regime info
}

// T5.2 — Regime Context for reporting
#[derive(Debug, Clone, Serialize)]
pub struct RegimeContext {
    pub atr_regime: String,          // "low", "normal", "high"
    pub slope_regime: String,        // "flat", "trending_up", "trending_down"
    pub session: String,             // UTC hour bucket: "asia", "london", "ny"
}

impl TradeSignal {
    pub fn new(
        symbol: String,
        timeframe: String,
        signal: SignalType,
        price: Decimal,
        score: i32,
        reasons: Vec<String>,
        regime: Option<RegimeContext>,
    ) -> Self {
        use uuid::Uuid;
        
        let confidence = score.clamp(0, 100) as u8;
        let confidence_tier = match confidence {
            80..=100 => "high",
            65..=79 => "medium",
            _ => "low",
        }.to_string();
        
        Self {
            signal_id: Uuid::new_v4().to_string(),
            engine_version: ENGINE_VERSION.to_string(),
            symbol,
            timeframe,
            signal,
            price,
            confidence,
            confidence_tier,
            timestamp: Utc::now(),
            reasons,
            regime_context: regime,
        }
    }
}

impl fmt::Display for TradeSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, 
            "SIGNAL [{}]: {} [{}] {:?} @ {} (Conf: {}% - {})", 
            &self.signal_id[..8], self.symbol, self.timeframe, self.signal, 
            self.price, self.confidence, self.confidence_tier
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
    
    // Liquidity-aware BOS tracking
    pub has_equal_highs: bool,       // Likidite havuzu: eşit high'lar
    pub has_equal_lows: bool,        // Likidite havuzu: eşit low'lar
    pub last_bos_displacement: bool, // Son BOS displacement ile mi oldu?
    pub bos_candle_range: Option<Decimal>, // BOS mumunun range'i
}

impl Default for MarketStructure {
    fn default() -> Self {
        Self {
            last_pivot_high: None,
            last_pivot_low: None,
            trend: TrendState::Neutral,
            bos_confirmed: false,
            has_equal_highs: false,
            has_equal_lows: false,
            last_bos_displacement: false,
            bos_candle_range: None,
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

