use rust_decimal::prelude::*;
use rust_decimal::Decimal;

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
    let left = highs[idx - 3..idx].iter().all(|&h| h < current);
    // Check 3 bars after
    let right = highs[idx + 1..=idx + 3].iter().all(|&h| h < current);

    left && right
}

pub fn is_pivot_low(lows: &[Decimal], idx: usize) -> bool {
    if idx < 3 || idx + 3 >= lows.len() {
        return false;
    }
    let current = lows[idx];
    // Check 3 bars before
    let left = lows[idx - 3..idx].iter().all(|&l| l > current);
    // Check 3 bars after
    let right = lows[idx + 1..=idx + 3].iter().all(|&l| l > current);

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
            }
            None => high - low,
        };

        self.prev_close = Some(close);

        match self.current_value {
            Some(prev_atr) => {
                // RMA (Wilder's Smoothing): (Prev * (period-1) + TR) / period
                // Equivalent to: Prev + alpha * (TR - Prev)
                let new_atr = prev_atr + self.alpha * (tr - prev_atr);
                self.current_value = Some(new_atr);
            }
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

#[derive(Debug, Clone)]
pub struct Kama {
    length: usize,
    fast: usize,
    slow: usize,
    pub current_value: Option<Decimal>,
    /// Efficiency Ratio: 0.0 = choppy/ranging, 1.0 = perfect trend
    pub er: Decimal,
    /// Smoothing Constant: low = KAMA barely moves (ranging), high = KAMA tracks price (trending)
    pub sc: Decimal,
    prev_close: Option<Decimal>,
    abs_changes: std::collections::VecDeque<Decimal>,
    closes: std::collections::VecDeque<Decimal>,
    sum_abs_change: Decimal,
}

impl Kama {
    pub fn new(length: usize, fast: usize, slow: usize) -> Self {
        Self {
            length,
            fast,
            slow,
            current_value: None,
            er: Decimal::ZERO,
            sc: Decimal::ZERO,
            prev_close: None,
            abs_changes: std::collections::VecDeque::new(),
            closes: std::collections::VecDeque::new(),
            sum_abs_change: Decimal::ZERO,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Decimal {
        if let Some(prev) = self.prev_close {
            let change = (close - prev).abs();
            self.abs_changes.push_back(change);
            self.sum_abs_change += change;
            if self.abs_changes.len() > self.length {
                if let Some(old) = self.abs_changes.pop_front() {
                    self.sum_abs_change -= old;
                }
            }
        }

        self.prev_close = Some(close);
        self.closes.push_back(close);
        if self.closes.len() > self.length + 1 {
            self.closes.pop_front();
        }

        let er = if self.closes.len() > self.length && !self.sum_abs_change.is_zero() {
            let first = *self.closes.front().unwrap();
            (close - first).abs() / self.sum_abs_change
        } else {
            Decimal::ZERO
        };

        let fast_sc = Decimal::from(2) / (Decimal::from(self.fast) + Decimal::ONE);
        let slow_sc = Decimal::from(2) / (Decimal::from(self.slow) + Decimal::ONE);
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

        // Expose ER and SC for quality gate filtering
        self.er = er;
        self.sc = sc;

        let kama = match self.current_value {
            Some(prev) => prev + sc * (close - prev),
            None => close,
        };

        self.current_value = Some(kama);
        kama
    }
}

#[derive(Debug, Clone)]
pub struct Macd {
    fast: Ema,
    slow: Ema,
    signal: Ema,
    pub macd_line: Option<Decimal>,
    pub signal_line: Option<Decimal>,
    pub hist: Option<Decimal>,
}

impl Macd {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast: Ema::new(fast),
            slow: Ema::new(slow),
            signal: Ema::new(signal),
            macd_line: None,
            signal_line: None,
            hist: None,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<(Decimal, Decimal, Decimal)> {
        let fast_val = self.fast.update(close);
        let slow_val = self.slow.update(close);
        let macd_line = fast_val - slow_val;
        let signal_line = self.signal.update(macd_line);
        let hist = macd_line - signal_line;

        self.macd_line = Some(macd_line);
        self.signal_line = Some(signal_line);
        self.hist = Some(hist);

        Some((macd_line, signal_line, hist))
    }
}

#[derive(Debug, Clone)]
pub struct Adx {
    period: usize,
    smoothing: usize,
    prev_high: Option<Decimal>,
    prev_low: Option<Decimal>,
    prev_close: Option<Decimal>,
    tr_sm: Option<Decimal>,
    plus_dm_sm: Option<Decimal>,
    minus_dm_sm: Option<Decimal>,
    adx_sm: Option<Decimal>,
    pub pdi: Option<Decimal>,
    pub mdi: Option<Decimal>,
    pub adx: Option<Decimal>,
}

impl Adx {
    pub fn new(period: usize, smoothing: usize) -> Self {
        Self {
            period,
            smoothing,
            prev_high: None,
            prev_low: None,
            prev_close: None,
            tr_sm: None,
            plus_dm_sm: None,
            minus_dm_sm: None,
            adx_sm: None,
            pdi: None,
            mdi: None,
            adx: None,
        }
    }

    pub fn update(
        &mut self,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> Option<(Decimal, Decimal, Decimal)> {
        let (tr, plus_dm, minus_dm) = match (self.prev_high, self.prev_low, self.prev_close) {
            (Some(prev_high), Some(prev_low), Some(prev_close)) => {
                let up_move = high - prev_high;
                let down_move = prev_low - low;

                let plus_dm = if up_move > down_move && up_move > Decimal::ZERO {
                    up_move
                } else {
                    Decimal::ZERO
                };

                let minus_dm = if down_move > up_move && down_move > Decimal::ZERO {
                    down_move
                } else {
                    Decimal::ZERO
                };

                let hl = high - low;
                let hc = (high - prev_close).abs();
                let lc = (low - prev_close).abs();
                let tr = hl.max(hc).max(lc);

                (tr, plus_dm, minus_dm)
            }
            _ => {
                let tr = high - low;
                (tr, Decimal::ZERO, Decimal::ZERO)
            }
        };

        self.prev_high = Some(high);
        self.prev_low = Some(low);
        self.prev_close = Some(close);

        let period = Decimal::from(self.period);
        let smoothing = Decimal::from(self.smoothing);

        self.tr_sm = Some(match self.tr_sm {
            Some(prev) => prev + (tr - prev) / period,
            None => tr,
        });

        self.plus_dm_sm = Some(match self.plus_dm_sm {
            Some(prev) => prev + (plus_dm - prev) / period,
            None => plus_dm,
        });

        self.minus_dm_sm = Some(match self.minus_dm_sm {
            Some(prev) => prev + (minus_dm - prev) / period,
            None => minus_dm,
        });

        let tr_sm = self.tr_sm.unwrap_or(Decimal::ZERO);
        if tr_sm.is_zero() {
            return None;
        }

        let pdi = Decimal::from(100) * self.plus_dm_sm.unwrap_or(Decimal::ZERO) / tr_sm;
        let mdi = Decimal::from(100) * self.minus_dm_sm.unwrap_or(Decimal::ZERO) / tr_sm;

        let denom = pdi + mdi;
        let dx = if denom.is_zero() {
            Decimal::ZERO
        } else {
            Decimal::from(100) * (pdi - mdi).abs() / denom
        };

        let adx = match self.adx_sm {
            Some(prev) => prev + (dx - prev) / smoothing,
            None => dx,
        };

        self.pdi = Some(pdi);
        self.mdi = Some(mdi);
        self.adx = Some(adx);
        self.adx_sm = Some(adx);

        Some((pdi, mdi, adx))
    }
}

#[derive(Debug, Clone)]
pub struct StochRsi {
    length: usize,
    k_period: usize,
    d_period: usize,
    rsi_window: std::collections::VecDeque<Decimal>,
    k_window: std::collections::VecDeque<Decimal>,
    d_window: std::collections::VecDeque<Decimal>,
    pub k: Option<Decimal>,
    pub d: Option<Decimal>,
}

impl StochRsi {
    pub fn new(length: usize, k_period: usize, d_period: usize) -> Self {
        Self {
            length,
            k_period,
            d_period,
            rsi_window: std::collections::VecDeque::new(),
            k_window: std::collections::VecDeque::new(),
            d_window: std::collections::VecDeque::new(),
            k: None,
            d: None,
        }
    }

    pub fn update(&mut self, rsi: Decimal) -> Option<(Decimal, Decimal)> {
        self.rsi_window.push_back(rsi);
        if self.rsi_window.len() > self.length {
            self.rsi_window.pop_front();
        }

        if self.rsi_window.len() < self.length {
            return None;
        }

        let min = self.rsi_window.iter().cloned().min().unwrap_or(rsi);
        let max = self.rsi_window.iter().cloned().max().unwrap_or(rsi);
        let denom = max - min;
        let stoch = if denom.is_zero() {
            Decimal::ZERO
        } else {
            (rsi - min) / denom * Decimal::from(100)
        };

        self.k_window.push_back(stoch);
        if self.k_window.len() > self.k_period {
            self.k_window.pop_front();
        }

        let k = if self.k_window.is_empty() {
            stoch
        } else {
            self.k_window.iter().cloned().sum::<Decimal>() / Decimal::from(self.k_window.len())
        };

        self.d_window.push_back(k);
        if self.d_window.len() > self.d_period {
            self.d_window.pop_front();
        }

        let d = if self.d_window.is_empty() {
            k
        } else {
            self.d_window.iter().cloned().sum::<Decimal>() / Decimal::from(self.d_window.len())
        };

        self.k = Some(k);
        self.d = Some(d);

        Some((k, d))
    }
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
            let gain = if change > Decimal::ZERO {
                change
            } else {
                Decimal::ZERO
            };
            let loss = if change < Decimal::ZERO {
                change.abs()
            } else {
                Decimal::ZERO
            };

            let (new_avg_gain, new_avg_loss) = match (self.avg_gain, self.avg_loss) {
                (Some(ag), Some(al)) => {
                    // Wilder's Smoothing: (Previous Avg * (n-1) + Current) / n
                    let period = Decimal::from(self.period);
                    let ag = (ag * (period - Decimal::ONE) + gain) / period;
                    let al = (al * (period - Decimal::ONE) + loss) / period;
                    (ag, al)
                }
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
    history
        .iter()
        .find(|(idx, _)| *idx == target_idx)
        .map(|(_, rsi)| *rsi)
}
