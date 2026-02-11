#!/usr/bin/env python3
"""
Bitcoin Hourly LSTM Signal Filter — train_lstm.py

Trains an LSTM model on bitcoin_prices_all_time.csv (hourly OHLCV data)
to predict whether a profitable trade opportunity exists within K bars.

Usage:
    python train_lstm.py                          # defaults
    python train_lstm.py --epochs 30 --lookback 72
    python train_lstm.py --csv bitcoin_prices_all_time.csv --epochs 20

Output:
    models/lstm_filter.pt      — PyTorch state dict
    models/lstm_meta.json      — metadata (threshold, features, normalization stats)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM signal filter from hourly BTC CSV")
    p.add_argument("--csv", default="bitcoin_prices_all_time.csv", help="Path to hourly OHLCV CSV")
    p.add_argument("--lookback", type=int, default=72, help="Sequence length (hours)")
    p.add_argument("--rr", type=float, default=1.5, help="Risk:Reward ratio")
    p.add_argument("--atr-sl-mult", type=float, default=1.5, help="ATR multiplier for stop-loss")
    p.add_argument("--k-min", type=int, default=6, help="Min forward bars to scan")
    p.add_argument("--k-max", type=int, default=36, help="Max forward bars to scan")
    p.add_argument("--epochs", type=int, default=25, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--min-preds", type=int, default=20, help="Min predictions for threshold eval")
    p.add_argument("--thresh-min", type=float, default=0.30, help="Threshold search min")
    p.add_argument("--thresh-max", type=float, default=0.80, help="Threshold search max")
    p.add_argument("--thresh-steps", type=int, default=11, help="Threshold search granularity")
    p.add_argument("--diagnostics", action="store_true", help="Print diagnostic info")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ─── Model (ONNX-compatible) ────────────────────────────────────────────────

class LSTMFilter(nn.Module):
    """LSTM signal quality filter — compatible with export_onnx.py."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.3 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ─── Feature Engineering ────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (0-1 normalized)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 1.0 - 1.0 / (1.0 + rs)
    return rsi.fillna(0.5)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """MACD - Signal difference, normalized by close."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    diff = (macd - signal) / close.replace(0, np.nan)
    return diff.fillna(0.0)


def compute_bollinger_position(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Position within Bollinger Bands (-1 lower, 0 mid, +1 upper)."""
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0).replace(0, np.nan)
    pos = (close - sma) / (num_std * std)
    return pos.clip(-3, 3).fillna(0.0)


def compute_kama_series(close: pd.Series, length: int = 10, fast: int = 2, slow: int = 20) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    change = close.diff().abs().fillna(0.0)
    sum_abs_change = change.rolling(length, min_periods=1).sum()
    er = (close - close.shift(length)).abs() / sum_abs_change.replace(0, np.nan)
    er = er.fillna(0.0)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = close.copy().astype(float)
    vals = kama.values.copy()
    sc_vals = sc.values
    for i in range(1, len(vals)):
        vals[i] = vals[i - 1] + sc_vals[i] * (close.iloc[i] - vals[i - 1])
    return pd.Series(vals, index=close.index, name="kama")


def compute_obv_slope(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV slope, normalized."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (volume * direction).cumsum()
    obv_ma = obv.rolling(period, min_periods=1).mean()
    slope = (obv - obv_ma) / obv.rolling(period, min_periods=1).std(ddof=0).replace(0, np.nan)
    return slope.fillna(0.0).clip(-5, 5)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 technical features from OHLCV data."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Returns ──
    log_return = np.log(close / close.shift()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return_2h = np.log(close / close.shift(2)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return_4h = np.log(close / close.shift(4)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return_12h = np.log(close / close.shift(12)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return_24h = np.log(close / close.shift(24)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ── Volatility / ATR ──
    rolling_vol_20 = log_return.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    atr_pct = atr_14 / close.replace(0, np.nan)
    return_over_atr = (log_return / atr_pct).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-10, 10)

    # ── Volatility regime ──
    atr_slow = atr_14.rolling(168, min_periods=1).mean()  # 1-week rolling
    vol_regime = (atr_14 / atr_slow.replace(0, np.nan)).fillna(1.0).clip(0, 5)

    # ── RSI ──
    rsi_14 = compute_rsi(close, 14)

    # ── MACD ──
    macd_signal_diff = compute_macd(close)

    # ── Bollinger ──
    bb_position = compute_bollinger_position(close)

    # ── KAMA ──
    kama = compute_kama_series(close)
    kama_slope = (kama - kama.shift()).fillna(0.0) / close.replace(0, np.nan)
    close_kama_over_atr = ((close - kama) / atr_14.replace(0, np.nan)).fillna(0.0).clip(-10, 10)

    # ── Volume ──
    vol_mean = volume.rolling(20, min_periods=1).mean()
    vol_std = volume.rolling(20, min_periods=1).std(ddof=0).replace(0, np.nan)
    volume_zscore = ((volume - vol_mean) / vol_std).fillna(0.0).clip(-5, 5)

    obv_slope = compute_obv_slope(close, volume)

    # ── VWAP ──
    dates = df.index.date if hasattr(df.index, "date") else pd.Series(df.index).dt.date.values
    vwap = (close * volume).groupby(dates).cumsum() / volume.groupby(dates).cumsum()
    vwap_distance = ((close - vwap) / vwap.replace(0, np.nan)).fillna(0.0).clip(-0.1, 0.1)

    # ── Time features ──
    hour = df.index.hour.astype(float) if hasattr(df.index, "hour") else 0.0
    hour_angle = 2.0 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_angle)
    hour_cos = np.cos(hour_angle)

    dow = df.index.dayofweek.astype(float) if hasattr(df.index, "dayofweek") else 0.0
    dow_angle = 2.0 * np.pi * dow / 7.0
    dow_sin = np.sin(dow_angle)
    dow_cos = np.cos(dow_angle)

    features = pd.DataFrame({
        "log_return": log_return,
        "return_2h": return_2h,
        "return_4h": return_4h,
        "return_12h": return_12h,
        "return_24h": return_24h,
        "rolling_vol_20": rolling_vol_20,
        "atr_14": atr_pct.fillna(0.0),  # normalized by close
        "return_over_atr": return_over_atr,
        "vol_regime": vol_regime,
        "rsi_14": rsi_14,
        "macd_signal_diff": macd_signal_diff,
        "bb_position": bb_position,
        "kama": (kama / close.replace(0, np.nan)).fillna(1.0),
        "kama_slope": kama_slope.fillna(0.0),
        "close_kama_over_atr": close_kama_over_atr,
        "volume_zscore": volume_zscore,
        "obv_slope": obv_slope,
        "vwap_distance": vwap_distance,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }, index=df.index)

    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features


FEATURE_COLUMNS = [
    "log_return",
    "return_2h",
    "return_4h",
    "return_12h",
    "return_24h",
    "rolling_vol_20",
    "atr_14",
    "return_over_atr",
    "vol_regime",
    "rsi_14",
    "macd_signal_diff",
    "bb_position",
    "kama",
    "kama_slope",
    "close_kama_over_atr",
    "volume_zscore",
    "obv_slope",
    "vwap_distance",
    "hour_sin",
    "hour_cos",
]


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load hourly OHLCV CSV and return cleaned DataFrame."""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Drop rows with zero close/volume
    df = df[(df["close"] > 0) & (df["volume"] > 0)]

    print(f"Loaded {len(df):,} hourly candles from {df.index.min()} to {df.index.max()}")
    return df


# ─── Label Generation ────────────────────────────────────────────────────────

def generate_labels(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    lookback: int,
    k: int,
    rr: float,
    atr_sl_mult: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) arrays from candles + features.

    For each bar i (where i >= lookback and future data exists):
      - SL distance = ATR_14 * atr_sl_mult
      - TP distance = SL distance * rr
      - Look forward k bars for TP/SL hit (TP hit → 1, else → 0)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    feat_vals = feats[FEATURE_COLUMNS].values
    n = len(df)

    # Pre-compute ATR for SL calculation
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]

    # Rolling ATR-14
    atr = np.zeros(n)
    atr[:14] = np.cumsum(tr[:14]) / np.arange(1, 15)
    for i in range(14, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14

    xs: List[np.ndarray] = []
    ys: List[int] = []

    start = max(lookback, 200)  # ensure indicators have warmed up
    end = n - k

    for i in range(start, end):
        sl_dist = atr[i] * atr_sl_mult
        if sl_dist <= 0 or close[i] <= 0:
            continue

        tp_dist = sl_dist * rr
        entry = close[i]

        # Check LONG: TP = entry + tp_dist, SL = entry - sl_dist
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist

        hit_tp = False
        hit_sl = False
        for j in range(i + 1, min(i + k + 1, n)):
            if high[j] >= tp_price:
                hit_tp = True
                break
            if low[j] <= sl_price:
                hit_sl = True
                break

        window = feat_vals[i - lookback + 1 : i + 1]
        if window.shape[0] != lookback:
            continue

        xs.append(window)
        ys.append(1 if hit_tp else 0)

    x_arr = np.stack(xs) if xs else np.empty((0, lookback, len(FEATURE_COLUMNS)))
    y_arr = np.array(ys, dtype=np.float32)
    return x_arr, y_arr


# ─── Train / Val / Test Split ────────────────────────────────────────────────

def time_split(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Walk-forward chronological split (no shuffling)."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def standardize(
    train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-8
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        {"mean": mean, "std": std},
    )


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_eval(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_preds: int,
    thresh_min: float,
    thresh_max: float,
    thresh_steps: int,
    diagnostics: bool,
) -> Tuple[float, float, float]:
    """Train model, find best threshold on val, evaluate on test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = x.shape[0]
    train_idx, val_idx, test_idx = time_split(n)

    train_x, val_x, test_x = x[train_idx], x[val_idx], x[test_idx]
    train_y, val_y, test_y = y[train_idx], y[val_idx], y[test_idx]

    train_x, val_x, test_x, stats = standardize(train_x, val_x, test_x)

    # Class balance weight
    pos_count = train_y.sum()
    neg_count = len(train_y) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    train_ds = SequenceDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LSTMFilter(train_x.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
    )
    loss_fn = nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bx)
            raw_loss = loss_fn(preds, by)
            # Apply class weights
            weights = torch.where(by == 1, pos_weight, 1.0)
            loss = (raw_loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Val loss ──
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(val_x, dtype=torch.float32, device=device)
            vy = torch.tensor(val_y, dtype=torch.float32, device=device)
            val_preds = model(vx)
            val_loss = loss_fn(val_preds, vy).mean().item()

        if diagnostics:
            print(f"  epoch {epoch + 1:02d}  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if diagnostics:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = model(torch.tensor(val_x, dtype=torch.float32, device=device)).cpu().numpy()
        test_probs = model(torch.tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy()

    if diagnostics:
        label_pos_rate = float(y.mean()) if n > 0 else 0.0
        print(f"  label_pos_rate={label_pos_rate:.4f}  "
              f"val_probs[min/med/max]={val_probs.min():.4f}/{np.median(val_probs):.4f}/{val_probs.max():.4f}")

    # Threshold search (maximize precision with minimum prediction count)
    best_precision = 0.0
    best_thresh = 0.5
    for thresh in np.linspace(thresh_min, thresh_max, max(thresh_steps, 2)):
        preds_bin = (val_probs >= thresh).astype(int)
        pred_count = int(preds_bin.sum())
        if pred_count < min_preds:
            continue
        true_pos = int(((preds_bin == 1) & (val_y == 1)).sum())
        precision = true_pos / max(pred_count, 1)
        # Also compute win rate adjusted by number of trades
        score = precision
        if score > best_precision:
            best_precision = score
            best_thresh = float(thresh)

    # Test evaluation
    test_preds_bin = (test_probs >= best_thresh).astype(int)
    test_true_pos = ((test_preds_bin == 1) & (test_y == 1)).sum()
    test_pred_count = test_preds_bin.sum()
    test_precision = float(test_true_pos / max(test_pred_count, 1))

    return best_precision, test_precision, best_thresh


def train_final_model(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    threshold: float,
) -> Tuple[LSTMFilter, Dict[str, np.ndarray], float]:
    """Train final model on train+val, evaluate on held-out test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = x.shape[0]
    train_idx, val_idx, test_idx = time_split(n)

    train_x = np.concatenate([x[train_idx], x[val_idx]], axis=0)
    train_y = np.concatenate([y[train_idx], y[val_idx]], axis=0)
    test_x = x[test_idx]
    test_y = y[test_idx]

    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-8
    train_x_norm = (train_x - mean) / std
    test_x_norm = (test_x - mean) / std

    # Class balance
    pos_count = train_y.sum()
    neg_count = len(train_y) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    train_ds = SequenceDataset(train_x_norm, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LSTMFilter(train_x_norm.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
    )
    loss_fn = nn.BCELoss(reduction="none")

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bx)
            raw_loss = loss_fn(preds, by)
            weights = torch.where(by == 1, pos_weight, 1.0)
            loss = (raw_loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

    model.eval()
    model_cpu = model.cpu()

    with torch.no_grad():
        test_probs = model_cpu(torch.tensor(test_x_norm, dtype=torch.float32)).numpy()

    test_preds_bin = (test_probs >= threshold).astype(int)
    test_true_pos = ((test_preds_bin == 1) & (test_y == 1)).sum()
    test_pred_count = test_preds_bin.sum()
    test_precision = float(test_true_pos / max(test_pred_count, 1))

    # Additional metrics
    test_total = len(test_y)
    test_pos = int(test_y.sum())
    test_recall = float(test_true_pos / max(test_pos, 1))

    print(f"  Final model — test_precision={test_precision:.3f}  "
          f"test_recall={test_recall:.3f}  "
          f"trades={int(test_pred_count)}/{test_total}  "
          f"wins={int(test_true_pos)}")

    stats = {"mean": mean, "std": std}
    return model_cpu, stats, test_precision


# ─── Main ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("  Bitcoin LSTM Signal Filter — Training")
    print("=" * 60)

    # ── Load data ──
    df = load_csv(args.csv)

    # ── Compute features ──
    print("Computing features...")
    feats = compute_features(df)
    print(f"Features: {len(FEATURE_COLUMNS)} columns, {len(feats):,} rows")

    # ── Search optimal K ──
    print(f"\nSearching optimal forward-bar K in [{args.k_min}, {args.k_max}]...")
    print("-" * 70)

    best_k: Optional[int] = None
    best_val_precision = 0.0
    best_test_precision = 0.0
    best_thresh = 0.5

    for k in range(args.k_min, args.k_max + 1, 3):  # Step by 3 for speed
        print(f"\n▸ Generating labels for k={k}...")
        x, y = generate_labels(df, feats, args.lookback, k, args.rr, args.atr_sl_mult)

        if x.shape[0] < 500:
            print(f"  Skip k={k}: only {x.shape[0]} samples")
            continue

        pos_rate = y.mean()
        print(f"  Samples: {x.shape[0]:,}  pos_rate={pos_rate:.3f}")

        val_prec, test_prec, thresh = train_and_eval(
            x, y,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
            args.min_preds,
            args.thresh_min,
            args.thresh_max,
            args.thresh_steps,
            args.diagnostics,
        )

        print(f"  k={k:02d}  val_precision={val_prec:.3f}  test_precision={test_prec:.3f}  threshold={thresh:.2f}")

        if val_prec > best_val_precision:
            best_val_precision = val_prec
            best_test_precision = test_prec
            best_k = k
            best_thresh = thresh

    if best_k is None:
        print("\n✗ No valid K found. Try adjusting parameters.")
        return

    print("\n" + "=" * 60)
    print(f"  Best K={best_k}  val_precision={best_val_precision:.3f}  "
          f"test_precision={best_test_precision:.3f}  threshold={best_thresh:.2f}")
    print("=" * 60)

    # ── Train final model ──
    print("\nTraining final model (train+val)...")
    x_final, y_final = generate_labels(df, feats, args.lookback, best_k, args.rr, args.atr_sl_mult)
    model, stats, final_test_precision = train_final_model(
        x_final, y_final, args.epochs, args.batch_size, args.lr, args.patience, best_thresh,
    )

    # ── Save ──
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "lstm_filter.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "best_k": best_k,
        "rr": args.rr,
        "atr_sl_mult": args.atr_sl_mult,
        "lookback": args.lookback,
        "threshold": best_thresh,
        "val_precision": best_val_precision,
        "test_precision": final_test_precision,
        "feature_columns": FEATURE_COLUMNS,
        "input_dim": len(FEATURE_COLUMNS),
        "hidden_dim": 128,
        "num_layers": 2,
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "csv_file": args.csv,
    }
    meta_path = models_dir / "lstm_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved model  → {model_path}")
    print(f"✓ Saved meta   → {meta_path}")
    print(f"\nRun 'python export_onnx.py' to generate ONNX for the Rust bot.")


if __name__ == "__main__":
    main()
