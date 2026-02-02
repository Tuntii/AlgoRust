use rust_decimal::Decimal;
use rust_decimal::prelude::*;

#[derive(Debug, Clone)]
pub struct Ema {
    pub period: usize,
    pub current_value: Option<Decimal>,
    k: Decimal,
}

impl Ema {
    pub fn new(period: usize) -> Self {
        let period_dec = Decimal::from(period);
        let two = Decimal::from(2);
        let k = two / (period_dec + Decimal::ONE);
        Self {
            period,
            current_value: None,
            k,
        }
    }

    // Calculates the next EMA value based on the previous close
    pub fn update(&mut self, price: Decimal) -> Decimal {
        match self.current_value {
            Some(prev) => {
                let new_val = (price - prev) * self.k + prev;
                self.current_value = Some(new_val);
                new_val
            }
            None => {
                self.current_value = Some(price);
                price
            }
        }
    }
}

// Pivot detection logic
// 3-bar lookback/lookahead: Needs context to determine validation
// We can't determine lookahead in real-time without latency, but the PRD says:
// "Pivot onayı 3 mum gecikmeli gelir — kabul edilen trade-off."
// This means we check index `i` when we are at index `i+3`.

pub fn is_pivot_high(highs: &[Decimal], idx: usize) -> bool {
    if idx < 3 || idx + 3 >= highs.len() {
        return false;
    }
    let current = highs[idx];
    // Check 3 bars before
    let left = highs[idx-3..idx].iter().all(|&h| h < current);
    // Check 3 bars after
    let right = highs[idx+1..=idx+3].iter().all(|&h| h < current);
    
    left && right
}

pub fn is_pivot_low(lows: &[Decimal], idx: usize) -> bool {
    if idx < 3 || idx + 3 >= lows.len() {
        return false;
    }
    let current = lows[idx];
    // Check 3 bars before
    let left = lows[idx-3..idx].iter().all(|&l| l > current);
    // Check 3 bars after
    let right = lows[idx+1..=idx+3].iter().all(|&l| l > current);
    
    left && right
}

#[derive(Debug, Clone)]
pub struct Atr {
    period: usize,
    prev_close: Option<Decimal>,
    pub current_value: Option<Decimal>,
    alpha: Decimal, // 1/period for Wilder's smoothing
}

impl Atr {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: None,
            current_value: None,
            alpha: Decimal::ONE / Decimal::from(period),
        }
    }

    pub fn update(&mut self, high: Decimal, low: Decimal, close: Decimal) -> Option<Decimal> {
        let tr = match self.prev_close {
            Some(prev) => {
                let hl = high - low;
                let hc = (high - prev).abs();
                let lc = (low - prev).abs();
                hl.max(hc).max(lc)
            },
            None => high - low,
        };

        self.prev_close = Some(close);

        match self.current_value {
            Some(prev_atr) => {
                // RMA (Wilder's Smoothing): (Prev * (period-1) + TR) / period
                // Equivalent to: Prev + alpha * (TR - Prev)
                let new_atr = prev_atr + self.alpha * (tr - prev_atr);
                self.current_value = Some(new_atr);
            },
            None => {
                // Seed with TR
                self.current_value = Some(tr);
            }
        }
        
        self.current_value
    }
}

#[derive(Debug, Clone)]
pub struct Rsi {
    pub period: usize,
    pub current_value: Option<Decimal>,
    prev_close: Option<Decimal>,
    avg_gain: Option<Decimal>,
    avg_loss: Option<Decimal>,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            current_value: None,
            prev_close: None,
            avg_gain: None,
            avg_loss: None,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<Decimal> {
        if let Some(prev) = self.prev_close {
            let change = close - prev;
            let gain = if change > Decimal::ZERO { change } else { Decimal::ZERO };
            let loss = if change < Decimal::ZERO { change.abs() } else { Decimal::ZERO };

            let (new_avg_gain, new_avg_loss) = match (self.avg_gain, self.avg_loss) {
                (Some(ag), Some(al)) => {
                    // Wilder's Smoothing: (Previous Avg * (n-1) + Current) / n
                    let period = Decimal::from(self.period);
                    let ag = (ag * (period - Decimal::ONE) + gain) / period;
                    let al = (al * (period - Decimal::ONE) + loss) / period;
                    (ag, al)
                },
                _ => (gain, loss), // Initial seeding (partially correct, smoothes out over time)
            };

            self.avg_gain = Some(new_avg_gain);
            self.avg_loss = Some(new_avg_loss);

            let rs = if new_avg_loss.is_zero() {
                Decimal::from(100)
            } else {
                new_avg_gain / new_avg_loss
            };

            let rsi = Decimal::from(100) - (Decimal::from(100) / (Decimal::ONE + rs));
            self.current_value = Some(rsi);
        }

        self.prev_close = Some(close);
        self.current_value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceType {
    Bullish, // Price LL, RSI HL
    Bearish, // Price HH, RSI LH
    None,
}

// Detect divergence between Price and RSI on Pivots
pub fn check_divergence(
    price_highs: &[(usize, Decimal)], // (index, price) - Pivot Highs
    price_lows: &[(usize, Decimal)],  // (index, price) - Pivot Lows
    rsi_history: &[(usize, Decimal)], // (index, rsi) - History aligned with pivots
) -> DivergenceType {
    // Bullish Divergence needs:
    // 1. Current Pivot Low < Previous Pivot Low (Price LL)
    // 2. Current RSI Low > Previous RSI Low (RSI HL)
    if price_lows.len() >= 2 {
        let (curr_idx, curr_price) = price_lows.last().unwrap();
        let (prev_idx, prev_price) = price_lows.get(price_lows.len() - 2).unwrap();
        
        // Find corresponding RSI values for these pivot candles
        let curr_rsi = find_rsi_at(rsi_history, *curr_idx);
        let prev_rsi = find_rsi_at(rsi_history, *prev_idx);
        
        if let (Some(c_rsi), Some(p_rsi)) = (curr_rsi, prev_rsi) {
            // Price Lower Low AND RSI Higher Low AND RSI is Oversold territory (<40 usually, but divergence can happen anywhere)
            // But strict divergence implies RSI shows strength.
            if curr_price < prev_price && c_rsi > p_rsi {
                // Filter: RSI shouldn't be too high for bullish div (e.g. < 50-60?)
                // Let's keep it pure divergence for now.
                return DivergenceType::Bullish;
            }
        }
    }

    // Bearish Divergence needs:
    // 1. Current Pivot High > Previous Pivot High (Price HH)
    // 2. Current RSI High < Previous RSI High (RSI LH)
    if price_highs.len() >= 2 {
        let (curr_idx, curr_price) = price_highs.last().unwrap();
        let (prev_idx, prev_price) = price_highs.get(price_highs.len() - 2).unwrap();
        
        let curr_rsi = find_rsi_at(rsi_history, *curr_idx);
        let prev_rsi = find_rsi_at(rsi_history, *prev_idx);
        
        if let (Some(c_rsi), Some(p_rsi)) = (curr_rsi, prev_rsi) {
            if curr_price > prev_price && c_rsi < p_rsi {
                return DivergenceType::Bearish;
            }
        }
    }

    DivergenceType::None
}

fn find_rsi_at(history: &[(usize, Decimal)], target_idx: usize) -> Option<Decimal> {
    history.iter().find(|(idx, _)| *idx == target_idx).map(|(_, rsi)| *rsi)
}

