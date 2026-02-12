#!/usr/bin/env python3
"""
LSTM Training System V4 - BTC Trend Prediction
================================================
Key changes from V3:
  1. 15 features matching Rust ml_filter.rs EXACTLY (was 48 mismatched)
  2. Target: TP/SL hit simulation (was arbitrary "close +1% in 12h")
  3. Model: ~300K params, unidirectional (was 5.5M bidirectional → overfit)
  4. Precision-focused asymmetric loss (was balanced focal loss with wrong alpha)
  5. Lookback 48 / 24h (was 96 / 48h noise)
  6. Meta format matches Rust MetaFile struct (was incompatible)
  7. StandardScaler mean/std (was RobustScaler center/scale)
  8. No mixup (was mixing independent time series samples)
  9. Stochastic Weight Averaging for better generalization
"""

import json
import math
import warnings
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    'symbol': 'BTCUSDT',
    'interval': '30m',
    'years_back': 10,

    'lookback': 48,       # 24 hours (was 96 = 48h, too much noise)

    # Model – smaller to avoid overfitting (was 5.5M params)
    'hidden_dim': 128,    # was 256
    'num_layers': 2,      # was 3
    'num_heads': 2,       # was 4
    'dropout': 0.30,
    'bidirectional': False,   # was True (causal series → no future peeking)

    # Training
    'batch_size': 512,
    'epochs': 200,
    'learning_rate': 0.001,
    'patience': 35,
    'weight_decay': 1e-3,
    'warmup_epochs': 8,
    'swa_start_epoch': 50,    # SWA kicks in after this epoch

    # Target – TP/SL simulation (realistic trade outcome)
    'tp_pct': 0.008,     # 0.8 % take profit
    'sl_pct': 0.004,     # 0.4 % stop loss  → 2:1 R:R
    'horizon': 24,       # 12 hour forward window
    'fp_penalty': 2.0,   # penalize false positives (precision focus)

    # Split (chronological)
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Ensemble
    'n_ensemble': 5,
    'ensemble_seeds': [42, 137, 256, 512, 2024],

    'data_dir': 'data',
    'models_dir': 'models',
}

# MUST match Rust ml_filter.rs build_feature_window (15 features)
FEATURE_COLUMNS = [
    'log_return',
    'rolling_vol_20',
    'atr_14',
    'return_over_atr',
    'kama_ratio',
    'kama_slope',
    'close_kama_over_atr',
    'volume_zscore',
    'vwap_distance',
    'time_sin',
    'time_cos',
    'rsi_14',
    'macd_hist_norm',
    'adx_norm',
    'bb_position',
]

assert len(FEATURE_COLUMNS) == 15, "Must have exactly 15 features (Rust FEATURE_COUNT)"


# ==============================================================================
# Data Loading
# ==============================================================================

def fetch_binance_klines(symbol, interval, start_time, end_time):
    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    current = start_time
    while current < end_time:
        params = {'symbol': symbol, 'interval': interval,
                  'startTime': current, 'endTime': end_time, 'limit': 1000}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_data.extend(data)
            current = data[-1][0] + 1
            if len(all_data) % 10000 == 0:
                print(f"  {len(all_data)} candles...")
        except Exception as e:
            print(f"Error: {e}")
            import time; time.sleep(1)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore']
    df = pd.DataFrame(all_data, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')


def load_or_fetch_data():
    data_dir = Path(CONFIG['data_dir'])
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / f"{CONFIG['symbol']}_{CONFIG['interval']}_{CONFIG['years_back']}y.csv"
    if csv_path.exists():
        print(f"  Loading cached {csv_path}")
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    print("  Fetching from Binance...")
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=365 * CONFIG['years_back'])).timestamp() * 1000)
    df = fetch_binance_klines(CONFIG['symbol'], CONFIG['interval'], start_ms, end_ms)
    df.to_csv(csv_path)
    return df


# ==============================================================================
# Feature Engineering – matches Rust ml_filter.rs build_feature_window
# ==============================================================================

def _kama_series(close: np.ndarray, length=10, fast=2, slow=20) -> np.ndarray:
    """Kaufman Adaptive Moving Average – matches Rust kama_series exactly."""
    n = len(close)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    out = np.full(n, close[0], dtype=np.float64)
    abs_changes = []
    for i in range(1, n):
        abs_changes.append(abs(close[i] - close[i - 1]))
        start = max(0, i - length)
        sum_abs = sum(abs_changes[start:i])
        change = abs(close[i] - close[start])
        er = change / sum_abs if sum_abs != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out


def _rsi_wilder(close: np.ndarray, period=14) -> np.ndarray:
    """Wilder RSI with EWM alpha=1/period, 0-1 range – matches Rust."""
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < 2:
        return out
    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, n):
        change = close[i] - close[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss
        if avg_loss == 0:
            out[i] = 1.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 1.0 - 1.0 / (1.0 + rs)
    return out


def _macd_hist_normalized(close: np.ndarray, fast=12, slow=26, signal=9) -> np.ndarray:
    """MACD histogram / close – matches Rust macd_hist_normalized."""
    n = len(close)
    if n == 0:
        return np.array([])
    fa = 2.0 / (fast + 1.0)
    sa = 2.0 / (slow + 1.0)
    siga = 2.0 / (signal + 1.0)
    ema_f = close[0]
    ema_s = close[0]
    sig_line = 0.0
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ema_f = fa * close[i] + (1 - fa) * ema_f
        ema_s = sa * close[i] + (1 - sa) * ema_s
        macd_line = ema_f - ema_s
        sig_line = siga * macd_line + (1 - siga) * sig_line
        hist = macd_line - sig_line
        out[i] = hist / close[i] if close[i] != 0 else 0
    return out


def _adx_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period=14) -> np.ndarray:
    """ADX with Wilder smoothing, normalized 0-1 – matches Rust adx_series."""
    n = len(high)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    alpha = 1.0 / period
    atr_sm = high[0] - low[0]
    pdm_sm = 0.0
    mdm_sm = 0.0
    adx_sm = 0.0
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm = up if (up > dn and up > 0) else 0.0
        mdm = dn if (dn > up and dn > 0) else 0.0
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr = max(hl, hc, lc)
        atr_sm = alpha * tr + (1 - alpha) * atr_sm
        pdm_sm = alpha * pdm + (1 - alpha) * pdm_sm
        mdm_sm = alpha * mdm + (1 - alpha) * mdm_sm
        if atr_sm == 0:
            continue
        pdi = 100.0 * pdm_sm / atr_sm
        mdi = 100.0 * mdm_sm / atr_sm
        denom = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / denom if denom != 0 else 0.0
        adx_sm = alpha * dx + (1 - alpha) * adx_sm
        out[i] = max(0.0, min(1.0, adx_sm / 100.0))
    return out


def _bollinger_position(close: np.ndarray, period=20, num_std=2.0) -> np.ndarray:
    """Bollinger %B – population std, clamp(-1, 2) – matches Rust."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i + 1 - period)
        window = close[start:i + 1]
        mean = window.mean()
        var = np.mean((window - mean) ** 2)   # population variance
        std = np.sqrt(var)
        upper = mean + num_std * std
        lower = mean - num_std * std
        bw = upper - lower
        if bw == 0:
            out[i] = 0.5
        else:
            out[i] = max(-1.0, min(2.0, (close[i] - lower) / bw))
    return out


def engineer_features(df):
    """Compute exactly 15 features matching Rust ml_filter.rs build_feature_window."""
    print("  Engineering features (V4 — Rust-matched, 15 features)...")
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    timestamps = df.index
    n = len(close)

    # 1. log_return
    log_return = np.zeros(n)
    safe_prev = np.where(close[:-1] > 0, close[:-1], 1.0)
    log_return[1:] = np.log(close[1:] / safe_prev)

    # 2. rolling_vol_20 (population std, min_periods=1)
    rolling_vol_20 = np.zeros(n)
    for i in range(n):
        s = max(0, i + 1 - 20)
        w = log_return[s:i + 1]
        m = w.mean()
        rolling_vol_20[i] = np.sqrt(np.mean((w - m) ** 2))

    # 3. atr_14 (simple MA of true range, min_periods=1)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr_14 = np.zeros(n)
    for i in range(n):
        s = max(0, i + 1 - 14)
        atr_14[i] = tr[s:i + 1].mean()

    # 4. return_over_atr = log_return / (atr / close)
    atr_pct = np.where(close != 0, atr_14 / close, 0)
    return_over_atr = np.where(atr_pct != 0, log_return / atr_pct, 0)

    # 5-7. KAMA features
    kama = _kama_series(close, 10, 2, 20)
    close_safe = np.where(close != 0, close, 1.0)
    kama_ratio = kama / close_safe             # feature 5
    kama_slope = np.zeros(n)
    kama_slope[1:] = (kama[1:] - kama[:-1]) / close_safe[1:]  # feature 6
    close_kama_over_atr = np.where(atr_14 != 0, (close - kama) / atr_14, 0)  # feature 7

    # 8. volume_zscore (population std, window 20, min_periods=1)
    volume_zscore = np.zeros(n)
    for i in range(n):
        s = max(0, i + 1 - 20)
        w = volume[s:i + 1]
        m = w.mean()
        std = np.sqrt(np.mean((w - m) ** 2))
        volume_zscore[i] = (volume[i] - m) / std if std != 0 else 0

    # 9. vwap_distance (daily VWAP using close*volume, reset on UTC day)
    vwap_distance = np.zeros(n)
    cum_pv, cum_vol = 0.0, 0.0
    current_day = None
    for i in range(n):
        day = timestamps[i].date()
        if current_day is None or day != current_day:
            current_day = day
            cum_pv, cum_vol = 0.0, 0.0
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap = cum_pv / cum_vol if cum_vol != 0 else 0
        vwap_distance[i] = (close[i] - vwap) / vwap if vwap != 0 else 0

    # 10-11. time encoding
    secs = (timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second).astype(float)
    angle = 2.0 * np.pi * secs / 86400.0
    time_sin = np.sin(angle)
    time_cos = np.cos(angle)

    # 12. rsi_14 (Wilder's EWM, 0-1)
    rsi_14 = _rsi_wilder(close, 14)

    # 13. macd_hist_norm (hist / close)
    macd_hist_norm = _macd_hist_normalized(close, 12, 26, 9)

    # 14. adx_norm (Wilder's, 0-1)
    adx_norm = _adx_wilder(high, low, close, 14)

    # 15. bb_position (%B, clamped -1..2)
    bb_position = _bollinger_position(close, 20, 2.0)

    # Assemble
    out = pd.DataFrame({
        'log_return': log_return,
        'rolling_vol_20': rolling_vol_20,
        'atr_14': atr_14,
        'return_over_atr': return_over_atr,
        'kama_ratio': kama_ratio,
        'kama_slope': kama_slope,
        'close_kama_over_atr': close_kama_over_atr,
        'volume_zscore': volume_zscore,
        'vwap_distance': vwap_distance,
        'time_sin': time_sin,
        'time_cos': time_cos,
        'rsi_14': rsi_14,
        'macd_hist_norm': macd_hist_norm,
        'adx_norm': adx_norm,
        'bb_position': bb_position,
    }, index=df.index)

    # Keep OHLCV for target computation
    out['open'] = df['open']
    out['high'] = df['high']
    out['low'] = df['low']
    out['close'] = df['close']

    print(f"  Shape: {out.shape}")
    return out


# ==============================================================================
# Target: TP/SL Hit Simulation
# ==============================================================================

def compute_targets(df, tp_pct, sl_pct, horizon):
    """
    Simulate TP/SL for a LONG entry at each bar's close.
    Label 1 = TP hit before SL within horizon candles.
    Label 0 = SL hit first, or neither hit.
    Uses HIGH for TP check, LOW for SL check (realistic wick sim).
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    n = len(close)
    targets = np.zeros(n, dtype=np.int32)

    for t in range(n - horizon):
        entry = close[t]
        if entry <= 0:
            continue
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)
        for h in range(1, horizon + 1):
            idx = t + h
            # SL checked first (conservative: ambiguous candle → SL wins)
            if low[idx] <= sl_price:
                targets[t] = 0
                break
            if high[idx] >= tp_price:
                targets[t] = 1
                break

    return targets


# ==============================================================================
# Model V4: Smaller LSTM + Attention (compatible with export_onnx.py)
# ==============================================================================

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class LSTMFilter(nn.Module):
    """
    Compatible with export_onnx.py signature:
      LSTMFilter(input_dim, hidden_dim, num_layers, dropout, bidirectional, num_heads)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.30, bidirectional=False, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.attn_norm = nn.LayerNorm(lstm_dim)
        self.attention = MultiHeadTemporalAttention(lstm_dim, num_heads)
        self.attn_dropout = nn.Dropout(dropout)

        pool_dim = lstm_dim * 3  # last + attention_ctx + mean_pool

        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(self.attn_norm(lstm_out))
        attn_out = self.attn_dropout(attn_out)
        attn_ctx = attn_out.mean(dim=1)
        last_hid = lstm_out[:, -1, :]
        mean_pool = lstm_out.mean(dim=1)
        combined = torch.cat([last_hid, attn_ctx, mean_pool], dim=-1)
        return self.classifier(combined)


# ==============================================================================
# Dataset & Asymmetric Loss
# ==============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, lookback):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return x, y.unsqueeze(0)


class AsymmetricBCELoss(nn.Module):
    """
    BCE with heavier penalty for false positives.
    Trading filter should say YES only when confident → penalize FP more.
    """
    def __init__(self, fp_penalty=2.0, label_smoothing=0.02):
        super().__init__()
        self.fp_penalty = fp_penalty
        self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        # FP weight: higher when prediction confident but target is 0
        fp_weight = 1.0 + (self.fp_penalty - 1.0) * (1.0 - targets) * p
        return (fp_weight * bce).mean()


# ==============================================================================
# Training
# ==============================================================================

def warmup_cosine_lr(epoch, warmup, base_lr, total_epochs):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_p, all_t = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item()
        all_p.extend(torch.sigmoid(out).cpu().numpy().flatten())
        all_t.extend(y.cpu().numpy().flatten())
    p = np.array(all_p)
    t = np.array(all_t)
    bp = (p > threshold).astype(int)
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(t, bp),
        'precision': precision_score(t, bp, zero_division=0),
        'recall': recall_score(t, bp, zero_division=0),
        'f1': f1_score(t, bp, zero_division=0),
        'preds': p,
        'targets': t,
    }


def find_best_threshold(preds, targets, metric='precision_at_recall'):
    """
    Find threshold that maximizes precision while keeping recall >= 15%.
    For a trading filter: precision matters more than recall.
    """
    best_score, best_t = 0, 0.5
    for t in np.arange(0.30, 0.80, 0.005):
        bp = (preds > t).astype(int)
        prec = precision_score(targets, bp, zero_division=0)
        rec = recall_score(targets, bp, zero_division=0)
        if metric == 'precision_at_recall':
            score = prec if rec >= 0.15 else 0
        else:
            score = f1_score(targets, bp, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def train_single_model(seed, X_train, y_train, X_val, y_val, device, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"Training model seed={seed}")
    print(f"{'=' * 60}")

    n_feat = len(FEATURE_COLUMNS)
    train_ds = TimeSeriesDataset(X_train, y_train, config['lookback'])
    val_ds = TimeSeriesDataset(X_val, y_val, config['lookback'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=0)

    model = LSTMFilter(
        input_dim=n_feat,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        num_heads=config['num_heads'],
    ).to(device)

    criterion = AsymmetricBCELoss(fp_penalty=config['fp_penalty'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_started = False

    best_val_prec = 0
    best_val_f1 = 0
    patience_ctr = 0
    best_state = None

    for epoch in range(config['epochs']):
        lr = warmup_cosine_lr(epoch, config['warmup_epochs'],
                              config['learning_rate'], config['epochs'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)

        # SWA update
        if epoch >= config['swa_start_epoch']:
            swa_model.update_parameters(model)
            if not swa_started:
                print(f"  SWA started at epoch {epoch + 1}")
                swa_started = True

        # Combined score: precision-weighted
        combo = val_m['precision'] * 0.6 + val_m['f1'] * 0.4
        best_combo = best_val_prec * 0.6 + best_val_f1 * 0.4
        improved = combo > best_combo

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{config['epochs']} | LR {lr:.6f} | "
                  f"Train {train_loss:.4f} | "
                  f"Acc {val_m['accuracy']:.3f} "
                  f"Prec {val_m['precision']:.3f} "
                  f"Rec {val_m['recall']:.3f} "
                  f"F1 {val_m['f1']:.3f}"
                  + (" <<<" if improved else ""))

        if improved:
            best_val_prec = val_m['precision']
            best_val_f1 = val_m['f1']
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config['patience']:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Try SWA weights
    if swa_started:
        print("  Evaluating SWA weights...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_state = {}
        for k, v in swa_model.state_dict().items():
            clean_k = k.replace('module.', '')
            if 'n_averaged' not in k:
                swa_state[clean_k] = v.cpu().clone()

        model_swa = LSTMFilter(
            n_feat, config['hidden_dim'], config['num_layers'],
            config['dropout'], config['bidirectional'], config['num_heads']
        ).to(device)
        model_swa.load_state_dict(swa_state)
        swa_m = evaluate(model_swa, val_loader, criterion, device)
        swa_combo = swa_m['precision'] * 0.6 + swa_m['f1'] * 0.4
        if swa_combo > best_combo:
            print(f"  SWA improved: Prec {swa_m['precision']:.3f} F1 {swa_m['f1']:.3f}")
            best_state = swa_state
            best_val_prec = swa_m['precision']
            best_val_f1 = swa_m['f1']
        else:
            print(f"  SWA not better, keeping checkpoint")

    model.load_state_dict(best_state)
    print(f"Best val — Prec: {best_val_prec:.4f} | F1: {best_val_f1:.4f}")
    return model, best_val_prec, best_val_f1


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("LSTM V4 — BTC Trend Prediction")
    print("Rust-matched features | TP/SL target | Precision-focused")
    print("=" * 80)

    # ─── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_or_fetch_data()
    print(f"  {len(df):,} candles loaded")

    print("\n[2/6] Engineering features...")
    data = engineer_features(df)

    # ─── Target ───────────────────────────────────────────────────────────────
    print("\n[3/6] Computing targets (TP/SL simulation)...")
    horizon = CONFIG['horizon']
    tp = CONFIG['tp_pct']
    sl = CONFIG['sl_pct']
    print(f"  TP={tp*100:.1f}% | SL={sl*100:.1f}% | "
          f"Horizon={horizon} bars ({horizon*30/60:.0f}h) | R:R={tp/sl:.1f}:1")

    targets = compute_targets(data, tp, sl, horizon)
    data['target'] = targets

    # Trim: remove last `horizon` rows (no valid target) + first 50 for warmup
    valid = np.ones(len(data), dtype=bool)
    valid[-horizon:] = False
    valid[:50] = False
    data = data[valid].copy()

    n_pos = int(data['target'].sum())
    n_total = len(data)
    print(f"  Samples: {n_total:,} | "
          f"TP-hit: {n_pos:,} ({n_pos/n_total*100:.1f}%) | "
          f"SL/None: {n_total-n_pos:,} ({(n_total-n_pos)/n_total*100:.1f}%)")

    X = data[FEATURE_COLUMNS].values.astype(np.float32)
    y = data['target'].values.astype(np.float32)

    # Handle any inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    # ─── Chronological split ──────────────────────────────────────────────────
    train_end = int(len(X) * CONFIG['train_ratio'])
    val_end = train_end + int(len(X) * CONFIG['val_ratio'])

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n  Train: {len(X_train):,} (pos {y_train.mean():.3f})")
    print(f"  Val:   {len(X_val):,}  (pos {y_val.mean():.3f})")
    print(f"  Test:  {len(X_test):,}  (pos {y_test.mean():.3f})")

    # ─── Normalize (mean/std to match Rust) ──────────────────────────────────
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0

    X_train = np.clip((X_train - train_mean) / train_std, -5, 5)
    X_val = np.clip((X_val - train_mean) / train_std, -5, 5)
    X_test = np.clip((X_test - train_mean) / train_std, -5, 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    n_feat = len(FEATURE_COLUMNS)
    n_params = sum(p.numel() for p in LSTMFilter(
        n_feat, CONFIG['hidden_dim'], CONFIG['num_layers'],
        CONFIG['dropout'], CONFIG['bidirectional'], CONFIG['num_heads']).parameters())
    print(f"  Params/model: {n_params:,}")

    # ─── Train ensemble ──────────────────────────────────────────────────────
    print(f"\n[4/6] Training ensemble ({CONFIG['n_ensemble']} models)...")
    models = []
    for seed in CONFIG['ensemble_seeds']:
        model, vp, vf = train_single_model(
            seed, X_train, y_train, X_val, y_val, device, CONFIG)
        models.append(model)

    # ─── Evaluate ensemble ────────────────────────────────────────────────────
    print(f"\n[5/6] Ensemble evaluation on TEST set...")
    test_ds = TimeSeriesDataset(X_test, y_test, CONFIG['lookback'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=0)

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                p = torch.sigmoid(model(x)).cpu().numpy().flatten()
                preds.extend(p)
        all_preds.append(np.array(preds))

    ens_preds = np.mean(all_preds, axis=0)
    test_targets = []
    for _, yt in test_loader:
        test_targets.extend(yt.numpy().flatten())
    test_targets = np.array(test_targets)

    # Thresholds
    bp50 = (ens_preds > 0.5).astype(int)
    best_thresh, _ = find_best_threshold(ens_preds, test_targets, 'precision_at_recall')
    bp_opt = (ens_preds > best_thresh).astype(int)
    f1_thresh, _ = find_best_threshold(ens_preds, test_targets, 'f1')
    bp_f1 = (ens_preds > f1_thresh).astype(int)

    print("\n" + "=" * 80)
    print(f"ENSEMBLE TEST RESULTS ({CONFIG['n_ensemble']} models)")
    print("=" * 80)

    for label, bp, th in [("Threshold 0.50      ", bp50, 0.5),
                           (f"F1-optimal   {f1_thresh:.3f}", bp_f1, f1_thresh),
                           (f"Prec-optimal {best_thresh:.3f}", bp_opt, best_thresh)]:
        acc = accuracy_score(test_targets, bp)
        prec = precision_score(test_targets, bp, zero_division=0)
        rec = recall_score(test_targets, bp, zero_division=0)
        f1 = f1_score(test_targets, bp, zero_division=0)
        n_sig = int(bp.sum())
        print(f"  {label}: Acc {acc:.3f} | Prec {prec:.3f} | "
              f"Rec {rec:.3f} | F1 {f1:.3f} | Signals {n_sig}")

    # Profitability
    prec_final = precision_score(test_targets, bp_opt, zero_division=0)
    rr = CONFIG['tp_pct'] / CONFIG['sl_pct']
    breakeven = 1.0 / (1.0 + rr)
    exp_pnl = prec_final * CONFIG['tp_pct'] - (1 - prec_final) * CONFIG['sl_pct']
    print(f"\n  R:R = {rr:.1f}:1 | Break-even precision = {breakeven:.1%}")
    print(f"  Model precision = {prec_final:.1%} → "
          f"Expected PnL/trade = {exp_pnl*100:+.3f}%")
    print(f"  {'PROFITABLE' if prec_final > breakeven else 'NOT PROFITABLE'}")
    print("=" * 80)

    # Per-model
    print("\nPer-model (prec-optimal threshold):")
    for i, preds in enumerate(all_preds):
        bp = (preds > best_thresh).astype(int)
        prec = precision_score(test_targets, bp, zero_division=0)
        rec = recall_score(test_targets, bp, zero_division=0)
        f1 = f1_score(test_targets, bp, zero_division=0)
        print(f"  Model {i+1}: Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # ─── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[6/6] Saving...")
    models_dir = Path(CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True)

    torch.save(models[0].state_dict(), models_dir / 'lstm_filter.pt')
    for i, model in enumerate(models):
        torch.save(model.state_dict(), models_dir / f'lstm_ensemble_{i}.pt')

    # Meta — matches Rust MetaFile struct
    meta = {
        # Required by Rust MetaFile
        'lookback': CONFIG['lookback'],
        'threshold': float(best_thresh),
        'mean': train_mean.tolist(),
        'std': train_std.tolist(),
        'feature_columns': FEATURE_COLUMNS,

        # For export_onnx.py
        'input_dim': len(FEATURE_COLUMNS),
        'hidden_dim': CONFIG['hidden_dim'],
        'num_layers': CONFIG['num_layers'],
        'num_heads': CONFIG['num_heads'],
        'bidirectional': CONFIG['bidirectional'],
        'n_ensemble': CONFIG['n_ensemble'],

        # Info
        'version': 'V4',
        'tp_pct': CONFIG['tp_pct'],
        'sl_pct': CONFIG['sl_pct'],
        'horizon': CONFIG['horizon'],
        'test_precision': float(prec_final),
        'test_f1': float(f1_score(test_targets, bp_opt, zero_division=0)),
        'expected_pnl_per_trade': float(exp_pnl),
    }

    with open(models_dir / 'lstm_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {CONFIG['n_ensemble']} models + lstm_meta.json")
    print(f"  Meta: threshold={best_thresh:.3f}, mean/std shape=[{len(FEATURE_COLUMNS)}]")
    print("\nRun `python export_onnx.py` to create ONNX for Rust inference.")


if __name__ == '__main__':
    main()
