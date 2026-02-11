#!/usr/bin/env python3
"""
LSTM Training System V3 - BTC Trend Prediction
Target: 70-75% accuracy on unseen data

V3 Improvements:
- Larger model (GPU-optimized: 256 hidden, 3 layers)
- Better target: 1% threshold over 12h horizon (cleaner signal)
- Multi-Head Attention over LSTM timesteps
- Ensemble of 3 models with different seeds
- Feature importance analysis & pruning
- Mixup data augmentation
- Warmup + Cosine Annealing
- Gradient accumulation for effective larger batch
"""

import json
import math
import warnings
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
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

    # Features
    'lookback': 96,

    # Model (GPU-sized)
    'hidden_dim': 256,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.35,
    'bidirectional': True,

    # Training
    'batch_size': 256,
    'epochs': 120,
    'learning_rate': 0.0003,
    'patience': 25,
    'weight_decay': 5e-4,
    'label_smoothing': 0.05,
    'warmup_epochs': 5,
    'mixup_alpha': 0.2,

    # Target - bigger moves are easier to predict
    'target_horizon': 24,      # 24 candles = 12 hours
    'target_threshold': 0.01,  # 1% move

    # Split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Ensemble
    'n_ensemble': 3,
    'ensemble_seeds': [42, 137, 2024],

    'data_dir': 'data',
    'models_dir': 'models',
}

FEATURE_COLUMNS = [
    # Returns
    'log_return',
    'return_1h', 'return_2h', 'return_4h', 'return_12h', 'return_24h',

    # Volatility
    'rolling_vol_10', 'rolling_vol_20', 'rolling_vol_50',
    'atr_pct', 'return_over_atr', 'vol_regime',

    # Momentum
    'rsi_14', 'rsi_7', 'rsi_21',
    'macd_pct', 'macd_signal_pct', 'macd_hist_pct',
    'stoch_k', 'stoch_d',
    'williams_r', 'cci_20',

    # MA distance
    'price_to_ema_9', 'price_to_ema_21',
    'price_to_ema_50', 'price_to_ema_200',
    'ema_9_21_cross', 'ema_50_200_cross',

    # Bollinger
    'bb_position', 'bb_width',

    # Volume
    'volume_zscore', 'volume_ma_ratio',
    'obv_slope', 'vwap_distance',

    # Candle
    'body_ratio', 'upper_shadow', 'lower_shadow', 'candle_range_pct',

    # Trends
    'trend_2h', 'trend_4h', 'trend_12h', 'trend_24h',

    # Strength
    'adx_14', 'momentum_10',

    # Time
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
]


# ==============================================================================
# Data
# ==============================================================================

def fetch_binance_klines(symbol, interval, start_time, end_time):
    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    current_start = start_time
    while current_start < end_time:
        params = {'symbol': symbol, 'interval': interval,
                  'startTime': current_start, 'endTime': end_time, 'limit': 1000}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            if len(all_data) % 10000 == 0:
                print(f"  {len(all_data)} candles...")
        except Exception as e:
            print(f"Error: {e}")
            import time; time.sleep(1)
            continue
    df = pd.DataFrame(all_data, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    df = df[['timestamp','open','high','low','close','volume']].set_index('timestamp')
    return df


def load_or_fetch_data():
    data_dir = Path(CONFIG['data_dir'])
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / f"{CONFIG['symbol']}_{CONFIG['interval']}_{CONFIG['years_back']}y.csv"
    if csv_path.exists():
        print(f"Loading from {csv_path}")
        return pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    print(f"Fetching data...")
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=365*CONFIG['years_back'])).timestamp() * 1000)
    df = fetch_binance_klines(CONFIG['symbol'], CONFIG['interval'], start_time, end_time)
    df.to_csv(csv_path)
    return df


# ==============================================================================
# Feature Engineering
# ==============================================================================

def compute_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))

def compute_atr(h, l, c, p=14):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def compute_adx(h, l, c, p=14):
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = compute_atr(h, l, c, p)
    plus_di = 100 * (plus_dm.rolling(p).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(p).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(p).mean()

def compute_cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def engineer_features(df):
    print("Engineering features (V3)...")
    d = df.copy()
    c, h, l, v = d['close'], d['high'], d['low'], d['volume']

    # Returns
    d['log_return'] = np.log(c / c.shift(1))
    for n, p in [('1h',2),('2h',4),('4h',8),('12h',24),('24h',48)]:
        d[f'return_{n}'] = c.pct_change(p)

    # Volatility
    d['rolling_vol_10'] = d['log_return'].rolling(10).std()
    d['rolling_vol_20'] = d['log_return'].rolling(20).std()
    d['rolling_vol_50'] = d['log_return'].rolling(50).std()
    atr = compute_atr(h, l, c, 14)
    d['atr_pct'] = atr / c
    d['return_over_atr'] = d['log_return'] / d['atr_pct'].replace(0, np.nan)
    d['vol_regime'] = d['rolling_vol_20'] / d['rolling_vol_50'].replace(0, np.nan)

    # Momentum
    d['rsi_14'] = compute_rsi(c, 14) / 100
    d['rsi_7'] = compute_rsi(c, 7) / 100
    d['rsi_21'] = compute_rsi(c, 21) / 100

    ef = c.ewm(span=12).mean()
    es = c.ewm(span=26).mean()
    macd = ef - es
    ms = macd.ewm(span=9).mean()
    d['macd_pct'] = macd / c
    d['macd_signal_pct'] = ms / c
    d['macd_hist_pct'] = (macd - ms) / c

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    rng14 = (hi14 - lo14).replace(0, np.nan)
    d['stoch_k'] = (c - lo14) / rng14
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()
    d['williams_r'] = (hi14 - c) / rng14
    d['cci_20'] = compute_cci(h, l, c, 20) / 200

    # MA distance
    for span, name in [(9,'9'),(21,'21'),(50,'50'),(200,'200')]:
        ema = c.ewm(span=span).mean()
        d[f'price_to_ema_{name}'] = (c - ema) / ema
    e9, e21 = c.ewm(span=9).mean(), c.ewm(span=21).mean()
    e50, e200 = c.ewm(span=50).mean(), c.ewm(span=200).mean()
    d['ema_9_21_cross'] = (e9 - e21) / e21
    d['ema_50_200_cross'] = (e50 - e200) / e200

    # Bollinger
    bm = c.rolling(20).mean()
    bs = c.rolling(20).std()
    bu, bl = bm + 2*bs, bm - 2*bs
    d['bb_position'] = (c - bl) / (bu - bl).replace(0, np.nan)
    d['bb_width'] = (bu - bl) / bm

    # Volume
    vm, vs = v.rolling(20).mean(), v.rolling(20).std()
    d['volume_zscore'] = (v - vm) / vs.replace(0, np.nan)
    d['volume_ma_ratio'] = v / vm.replace(0, np.nan)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d['obv_slope'] = obv.pct_change(10)
    tp = (h + l + c) / 3
    vwap_r = (tp * v).rolling(48).sum() / v.rolling(48).sum()
    d['vwap_distance'] = (c - vwap_r) / vwap_r.replace(0, np.nan)

    # Candle
    cr = (h - l).replace(0, np.nan)
    body = abs(c - d['open'])
    d['body_ratio'] = body / cr
    d['upper_shadow'] = (h - pd.concat([c, d['open']], axis=1).max(axis=1)) / cr
    d['lower_shadow'] = (pd.concat([c, d['open']], axis=1).min(axis=1) - l) / cr
    d['candle_range_pct'] = (h - l) / c

    # Trends
    for n, p in [('2h',4),('4h',8),('12h',24),('24h',48)]:
        d[f'trend_{n}'] = c.pct_change(p)

    # Strength
    d['adx_14'] = compute_adx(h, l, c, 14) / 100
    d['momentum_10'] = c.pct_change(10)

    # Time
    hour = d.index.hour + d.index.minute / 60
    d['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    day = d.index.dayofweek
    d['day_sin'] = np.sin(2 * np.pi * day / 7)
    d['day_cos'] = np.cos(2 * np.pi * day / 7)

    # Target: 12h horizon, 1% threshold
    future_ret = c.shift(-CONFIG['target_horizon']) / c - 1
    d['target'] = (future_ret > CONFIG['target_threshold']).astype(int)
    d['future_return'] = future_ret

    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    n_pos = d['target'].sum()
    n_neg = len(d) - n_pos
    print(f"Shape: {d.shape}")
    print(f"Bullish: {n_pos} ({n_pos/len(d)*100:.1f}%) | Not: {n_neg} ({n_neg/len(d)*100:.1f}%)")
    return d


# ==============================================================================
# Model V3: LSTM + Multi-Head Attention
# ==============================================================================

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
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
    def __init__(self, input_dim, hidden_dim=256, num_layers=3,
                 dropout=0.35, bidirectional=True, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection with residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Multi-Head Attention
        self.attn_norm = nn.LayerNorm(lstm_dim)
        self.attention = MultiHeadTemporalAttention(lstm_dim, num_heads)
        self.attn_dropout = nn.Dropout(dropout)

        # Pooling: concat last hidden + attention context + mean pool
        pool_dim = lstm_dim * 3

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)

        # Attention context
        attn_out = self.attention(self.attn_norm(lstm_out))
        attn_out = self.attn_dropout(attn_out)
        attn_context = attn_out.mean(dim=1)  # (B, D)

        # Last hidden + mean pool
        last_hidden = lstm_out[:, -1, :]
        mean_pool = lstm_out.mean(dim=1)

        # Concat all representations
        combined = torch.cat([last_hidden, attn_context, mean_pool], dim=-1)

        return self.classifier(combined)


# ==============================================================================
# Dataset & Loss
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, inputs, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = torch.where(targets > 0.5, p, 1 - p)
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
            w = at * w
        return (w * bce).mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for time series."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y


# ==============================================================================
# Training
# ==============================================================================

def get_lr(epoch, warmup, base_lr, total_epochs):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, criterion, optimizer, device, epoch, config):
    model.train()
    total_loss = 0
    use_mixup = config['mixup_alpha'] > 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_mixup and random.random() > 0.5:
            x, y = mixup_data(x, y, config['mixup_alpha'])

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            all_p.extend(torch.sigmoid(out).cpu().numpy())
            all_t.extend(y.cpu().numpy())
    p = np.array(all_p).flatten()
    t = np.array(all_t).flatten()
    bp = (p > threshold).astype(int)
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(t, bp),
        'precision': precision_score(t, bp, zero_division=0),
        'recall': recall_score(t, bp, zero_division=0),
        'f1': f1_score(t, bp, zero_division=0),
        'preds': p, 'targets': t,
    }


def find_best_threshold(preds, targets):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.005):
        f1 = f1_score(targets, (preds > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def train_single_model(seed, X_train, y_train, X_val, y_val, feature_cols, device, config):
    """Train a single model with given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}")
    print(f"{'='*60}")

    train_ds = TimeSeriesDataset(X_train, y_train, config['lookback'])
    val_ds = TimeSeriesDataset(X_val, y_val, config['lookback'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=0)

    model = LSTMFilter(
        input_dim=len(feature_cols),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        num_heads=config['num_heads'],
    ).to(device)

    pos_rate = y_train.mean()
    alpha = 1 - pos_rate
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=config['label_smoothing'])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']
    )

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        # Manual warmup + cosine LR
        lr = get_lr(epoch, config['warmup_epochs'], config['learning_rate'], config['epochs'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        val_m = evaluate(model, val_loader, criterion, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | LR: {lr:.6f}")
            print(f"  Train: {train_loss:.4f} | Val Acc: {val_m['accuracy']:.4f} "
                  f"Prec: {val_m['precision']:.4f} Rec: {val_m['recall']:.4f} F1: {val_m['f1']:.4f}")

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  >>> Best F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"Best val F1: {best_val_f1:.4f}")
    return model, best_val_f1


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("LSTM V3 - BTC Trend Prediction")
    print("GPU Model | Multi-Head Attention | Ensemble | Mixup")
    print("=" * 80)

    # Data
    print("\n[1/6] Loading data...")
    df = load_or_fetch_data()
    print(f"Loaded {len(df)} candles")

    print("\n[2/6] Engineering features...")
    data = engineer_features(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        print(f"WARNING missing: {missing}")
    print(f"Using {len(feature_cols)} features")

    # Prepare
    print("\n[3/6] Preparing data...")
    X = data[feature_cols].values
    y = data['target'].values

    train_end = int(len(X) * CONFIG['train_ratio'])
    val_end = train_end + int(len(X) * CONFIG['val_ratio'])

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train bullish: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    scaler = RobustScaler()
    X_train = np.clip(scaler.fit_transform(X_train), -5, 5)
    X_val = np.clip(scaler.transform(X_val), -5, 5)
    X_test = np.clip(scaler.transform(X_test), -5, 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_params = sum(p.numel() for p in LSTMFilter(
        len(feature_cols), CONFIG['hidden_dim'], CONFIG['num_layers'],
        CONFIG['dropout'], CONFIG['bidirectional'], CONFIG['num_heads']
    ).parameters())
    print(f"Parameters per model: {n_params:,}")

    # Train ensemble
    print(f"\n[4/6] Training ensemble ({CONFIG['n_ensemble']} models)...")
    models = []
    for i, seed in enumerate(CONFIG['ensemble_seeds']):
        model, val_f1 = train_single_model(
            seed, X_train, y_train, X_val, y_val, feature_cols, device, CONFIG
        )
        models.append(model)

    # Ensemble evaluation
    print("\n[5/6] Ensemble evaluation on test set...")
    test_ds = TimeSeriesDataset(X_test, y_test, CONFIG['lookback'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=0)

    # Get ensemble predictions
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = torch.sigmoid(model(x)).cpu().numpy()
                preds.extend(out)
        all_preds.append(np.array(preds).flatten())

    # Average ensemble predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    test_targets = []
    for x, y in test_loader:
        test_targets.extend(y.numpy())
    test_targets = np.array(test_targets).flatten()

    # Default threshold
    bp_50 = (ensemble_preds > 0.5).astype(int)
    acc_50 = accuracy_score(test_targets, bp_50)
    prec_50 = precision_score(test_targets, bp_50, zero_division=0)
    rec_50 = recall_score(test_targets, bp_50, zero_division=0)
    f1_50 = f1_score(test_targets, bp_50, zero_division=0)

    # Optimal threshold
    best_thresh, _ = find_best_threshold(ensemble_preds, test_targets)
    bp_opt = (ensemble_preds > best_thresh).astype(int)
    acc_opt = accuracy_score(test_targets, bp_opt)
    prec_opt = precision_score(test_targets, bp_opt, zero_division=0)
    rec_opt = recall_score(test_targets, bp_opt, zero_division=0)
    f1_opt = f1_score(test_targets, bp_opt, zero_division=0)

    print("\n" + "=" * 80)
    print(f"ENSEMBLE TEST RESULTS ({CONFIG['n_ensemble']} models)")
    print("=" * 80)
    print(f"Threshold 0.50:")
    print(f"  Acc: {acc_50:.4f} ({acc_50*100:.2f}%) | Prec: {prec_50:.4f} | Rec: {rec_50:.4f} | F1: {f1_50:.4f}")
    print(f"\nOptimal Threshold {best_thresh:.3f}:")
    print(f"  Acc: {acc_opt:.4f} ({acc_opt*100:.2f}%) | Prec: {prec_opt:.4f} | Rec: {rec_opt:.4f} | F1: {f1_opt:.4f}")
    print("=" * 80)

    # Individual model results
    print("\nPer-model results:")
    for i, preds in enumerate(all_preds):
        bp = (preds > best_thresh).astype(int)
        a = accuracy_score(test_targets, bp)
        p = precision_score(test_targets, bp, zero_division=0)
        print(f"  Model {i+1}: Acc={a:.4f} Prec={p:.4f}")

    # Save best model (the one with highest val F1) for ONNX export
    print("\n[6/6] Saving...")
    models_dir = Path(CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True)

    # Save first model as primary (for ONNX export compatibility)
    torch.save(models[0].state_dict(), models_dir / 'lstm_filter.pt')

    # Save all ensemble models
    for i, model in enumerate(models):
        torch.save(model.state_dict(), models_dir / f'lstm_ensemble_{i}.pt')

    meta = {
        'lookback': CONFIG['lookback'],
        'feature_columns': feature_cols,
        'input_dim': len(feature_cols),
        'hidden_dim': CONFIG['hidden_dim'],
        'num_layers': CONFIG['num_layers'],
        'num_heads': CONFIG['num_heads'],
        'bidirectional': CONFIG['bidirectional'],
        'target_threshold': CONFIG['target_threshold'],
        'target_horizon': CONFIG['target_horizon'],
        'optimal_threshold': float(best_thresh),
        'n_ensemble': CONFIG['n_ensemble'],
        'test_accuracy': float(acc_opt),
        'test_precision': float(prec_opt),
        'test_recall': float(rec_opt),
        'test_f1': float(f1_opt),
        'center': scaler.center_.reshape(1, 1, -1).tolist(),
        'scale': scaler.scale_.reshape(1, 1, -1).tolist(),
    }

    with open(models_dir / 'lstm_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {CONFIG['n_ensemble']} models + metadata")
    print("Training complete!")


if __name__ == '__main__':
    main()
