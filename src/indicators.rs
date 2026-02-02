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
