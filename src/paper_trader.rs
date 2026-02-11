use crate::state::SymbolContext;
use crate::types::{SignalType, TradeSignal};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTrader {
    pub initial_balance: Decimal,
    pub current_balance: Decimal,
    pub positions: HashMap<String, Position>,
    pub closed_trades: Vec<Trade>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: PositionSide,
    pub entry_price: Decimal,
    pub quantity: Decimal,
    pub stop_loss: Decimal,
    pub take_profit: Decimal,
    pub opened_at: DateTime<Utc>,
    pub signal_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub side: PositionSide,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub quantity: Decimal,
    pub pnl: Decimal,
    pub pnl_percent: Decimal,
    pub exit_reason: ExitReason,
    pub opened_at: DateTime<Utc>,
    pub closed_at: DateTime<Utc>,
    pub signal_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
    Manual,
}

impl PaperTrader {
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            initial_balance,
            current_balance: initial_balance,
            positions: HashMap::new(),
            closed_trades: Vec::new(),
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_pnl: Decimal::ZERO,
        }
    }

    /// Load paper trader state from JSON file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).context("Failed to read paper trader state file")?;
        let trader: PaperTrader =
            serde_json::from_str(&content).context("Failed to parse paper trader state")?;
        Ok(trader)
    }

    /// Save paper trader state to JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json =
            serde_json::to_string_pretty(self).context("Failed to serialize paper trader state")?;
        fs::write(path, json).context("Failed to write paper trader state file")?;
        Ok(())
    }

    /// Calculate position size based on risk percentage
    pub fn calculate_position_size(
        &self,
        entry_price: Decimal,
        sl_price: Decimal,
        risk_percent: Decimal,
        confidence_multiplier: Decimal,
    ) -> Decimal {
        // Risk amount in USD
        let risk_amount = self.current_balance * risk_percent;

        // Risk per unit (distance from entry to SL)
        let risk_per_unit = (entry_price - sl_price).abs();

        if risk_per_unit == Decimal::ZERO {
            warn!("⚠️ Risk per unit is zero, using minimum position size");
            return Decimal::from_f64(0.001).unwrap();
        }

        // Base position size
        let base_size = risk_amount / risk_per_unit;

        // Apply confidence multiplier
        let final_size = base_size * confidence_multiplier;

        // Cap at 10% of portfolio value
        let max_position_value = self.current_balance * Decimal::from_f64(0.10).unwrap();
        let position_value = final_size * entry_price;

        if position_value > max_position_value {
            warn!("⚠️ Position size exceeds 10% of portfolio, capping at max");
            max_position_value / entry_price
        } else {
            final_size
        }
    }

    /// Open a new position from signal
    pub fn open_position(
        &mut self,
        signal: &TradeSignal,
        ctx: &SymbolContext,
        risk_percent: Decimal,
        confidence_multiplier: Decimal,
    ) -> Result<()> {
        let entry_price = signal.price;
        let (sl_price, tp_price) = calculate_sl_tp(signal, ctx, entry_price);

        // Calculate position size
        let quantity = self.calculate_position_size(
            entry_price,
            sl_price,
            risk_percent,
            confidence_multiplier,
        );

        // Check if we have enough balance
        let required_balance = quantity * entry_price;
        if required_balance > self.current_balance {
            warn!(
                "⚠️ Insufficient balance: required ${}, available ${}",
                required_balance, self.current_balance
            );
            return Ok(());
        }

        let side = match signal.signal {
            SignalType::LONG => PositionSide::Long,
            SignalType::SHORT => PositionSide::Short,
        };

        let position = Position {
            symbol: signal.symbol.clone(),
            side,
            entry_price,
            quantity,
            stop_loss: sl_price,
            take_profit: tp_price,
            opened_at: Utc::now(),
            signal_id: signal.signal_id.clone(),
        };

        // Deduct balance (for both long and short - margin simulation)
        self.current_balance -= required_balance;

        let open_count = self
            .positions
            .values()
            .filter(|p| p.symbol == signal.symbol)
            .count();
        info!(
            "📈 OPENED POSITION: {} {} @ ${} (pos #{} for {})",
            signal.signal,
            signal.symbol,
            entry_price,
            open_count + 1,
            signal.symbol
        );
        info!("   Qty: {}, SL: ${}, TP: ${}", quantity, sl_price, tp_price);
        info!(
            "   Balance: ${} -> ${}",
            self.current_balance + required_balance,
            self.current_balance
        );

        self.positions.insert(signal.signal_id.clone(), position);
        Ok(())
    }

    /// Update all positions with current price (check SL/TP)
    pub fn update_positions(&mut self, symbol: &str, current_price: Decimal) -> Result<()> {
        // Collect signal_ids for positions of this symbol that need closing
        let mut to_close: Vec<(String, ExitReason)> = Vec::new();

        for (signal_id, position) in self.positions.iter() {
            if position.symbol != symbol {
                continue;
            }
            let sig_tag = &signal_id[..signal_id.len().min(8)];
            match position.side {
                PositionSide::Long => {
                    if current_price <= position.stop_loss {
                        info!(
                            "🛑 STOP LOSS HIT: {} @ ${} (sig: {})",
                            symbol, current_price, sig_tag
                        );
                        to_close.push((signal_id.clone(), ExitReason::StopLoss));
                    } else if current_price >= position.take_profit {
                        info!(
                            "🎯 TAKE PROFIT HIT: {} @ ${} (sig: {})",
                            symbol, current_price, sig_tag
                        );
                        to_close.push((signal_id.clone(), ExitReason::TakeProfit));
                    }
                }
                PositionSide::Short => {
                    if current_price >= position.stop_loss {
                        info!(
                            "🛑 STOP LOSS HIT: {} @ ${} (sig: {})",
                            symbol, current_price, sig_tag
                        );
                        to_close.push((signal_id.clone(), ExitReason::StopLoss));
                    } else if current_price <= position.take_profit {
                        info!(
                            "🎯 TAKE PROFIT HIT: {} @ ${} (sig: {})",
                            symbol, current_price, sig_tag
                        );
                        to_close.push((signal_id.clone(), ExitReason::TakeProfit));
                    }
                }
            }
        }

        for (signal_id, reason) in to_close {
            self.close_position(&signal_id, current_price, reason)?;
        }

        Ok(())
    }

    /// Close position and calculate P&L
    fn close_position(
        &mut self,
        signal_id: &str,
        exit_price: Decimal,
        reason: ExitReason,
    ) -> Result<()> {
        let position = self
            .positions
            .remove(signal_id)
            .context("Position not found")?;

        // Calculate P&L
        let pnl = match position.side {
            PositionSide::Long => {
                // Long: profit if price goes up
                (exit_price - position.entry_price) * position.quantity
            }
            PositionSide::Short => {
                // Short: profit if price goes down
                (position.entry_price - exit_price) * position.quantity
            }
        };

        let pnl_percent = (pnl / (position.entry_price * position.quantity)) * Decimal::from(100);

        // Return initial margin + P&L
        let return_amount = (position.entry_price * position.quantity) + pnl;
        self.current_balance += return_amount;

        // Update statistics
        self.total_trades += 1;
        self.total_pnl += pnl;

        if pnl > Decimal::ZERO {
            self.winning_trades += 1;
        } else if pnl < Decimal::ZERO {
            self.losing_trades += 1;
        }

        let trade = Trade {
            symbol: position.symbol.clone(),
            side: position.side.clone(),
            entry_price: position.entry_price,
            exit_price,
            quantity: position.quantity,
            pnl,
            pnl_percent,
            exit_reason: reason,
            opened_at: position.opened_at,
            closed_at: Utc::now(),
            signal_id: position.signal_id,
        };

        let sig_short = &signal_id[..signal_id.len().min(8)];
        let pnl_emoji = if pnl > Decimal::ZERO { "💰" } else { "📉" };
        info!(
            "{} CLOSED POSITION: {} @ ${} (sig: {})",
            pnl_emoji, &position.symbol, exit_price, sig_short
        );
        info!("   P&L: ${} ({:.2}%)", pnl, pnl_percent);
        info!("   Balance: ${}", self.current_balance);

        self.closed_trades.push(trade);
        Ok(())
    }

    /// Print current portfolio status
    pub fn print_status(&self) {
        info!("💼 PORTFOLIO STATUS:");
        info!(
            "   Balance: ${} (Initial: ${})",
            self.current_balance, self.initial_balance
        );
        info!(
            "   Total P&L: ${} ({:.2}%)",
            self.total_pnl,
            (self.total_pnl / self.initial_balance) * Decimal::from(100)
        );
        info!("   Open Positions: {}", self.positions.len());
        info!(
            "   Total Trades: {} (W: {}, L: {})",
            self.total_trades, self.winning_trades, self.losing_trades
        );

        if self.total_trades > 0 {
            let win_rate = (Decimal::from(self.winning_trades) / Decimal::from(self.total_trades))
                * Decimal::from(100);
            info!("   Win Rate: {:.1}%", win_rate);
        }
    }
}

/// Calculate SL/TP based on pivots (same logic as alpaca.rs)
fn calculate_sl_tp(
    signal: &TradeSignal,
    ctx: &SymbolContext,
    entry: Decimal,
) -> (Decimal, Decimal) {
    let default_rr = Decimal::from_f64(1.5).unwrap();
    let min_profit_pct = Decimal::from_f64(0.005).unwrap();

    match signal.signal {
        SignalType::LONG => {
            let sl = ctx
                .structure
                .last_pivot_low
                .unwrap_or(entry * Decimal::from_f64(0.99).unwrap());

            let safe_sl = if (entry - sl) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(0.995).unwrap()
            } else {
                sl
            };

            let risk = entry - safe_sl;

            let mut target_tp = None;
            let mut best_tp = Decimal::MAX;

            for &pivot in &ctx.pivot_high_history {
                if pivot > entry * (Decimal::ONE + min_profit_pct) {
                    if pivot < best_tp {
                        best_tp = pivot;
                        target_tp = Some(pivot);
                    }
                }
            }

            let tp = target_tp.unwrap_or_else(|| entry + (risk * default_rr));
            (safe_sl, tp)
        }
        SignalType::SHORT => {
            let sl = ctx
                .structure
                .last_pivot_high
                .unwrap_or(entry * Decimal::from_f64(1.01).unwrap());

            let safe_sl = if (sl - entry) / entry < Decimal::from_f64(0.002).unwrap() {
                entry * Decimal::from_f64(1.005).unwrap()
            } else {
                sl
            };

            let risk = safe_sl - entry;

            let mut target_tp = None;
            let mut best_tp = Decimal::MIN;

            for &pivot in &ctx.pivot_low_history {
                if pivot < entry * (Decimal::ONE - min_profit_pct) {
                    if pivot > best_tp {
                        best_tp = pivot;
                        target_tp = Some(pivot);
                    }
                }
            }

            let tp = target_tp.unwrap_or_else(|| entry - (risk * default_rr));
            (safe_sl, tp)
        }
    }
}
