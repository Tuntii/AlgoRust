"""
==============================================================================
Precision Confluence Backtest Sandbox
==============================================================================
SuperKAMA sinyallerini Confluence Score filtresiyle test eder.
Hedef: Az sinyal, yüksek isabetlilik (winrate > %70)

Mevcut Rust bottan bire bir port edilmiş SuperKAMA+ATR mantığı.
Confluence katmanları üzerine grid search ile optimal eşik bulunur.

Kullanım:
  python sandbox/backtest_precision.py
  python sandbox/backtest_precision.py --symbol ETHUSDT --tf 15m --days 365
  python sandbox/backtest_precision.py --download --symbol BTCUSDT --tf 1m --days 180

Parametreler (argümanlar olmadan da çalışır):
  --symbol  : ETHUSDT | BTCUSDT | SOLUSDT  (default: ETHUSDT)
  --tf      : 1m | 5m | 15m | 1h          (default: 15m)
  --days    : Kaç günlük backtest          (default: 365)
  --download: Binance'tan veri çek         (default: cache kullan)
  --grid    : Confluence grid search yap   (default: True)
  --min_score: Min confluence skoru        (default: grid search)
  --sl_atr  : SL ATR çarpanı              (default: grid search)
  --tp_atr  : TP ATR çarpanı              (default: grid search)
  --no_ml   : ML filtresini devre dışı bırak
==============================================================================
"""

import argparse
import sys
import os
import json
import math
import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ──────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pandas ve numpy gerekli. Kurulum: pip install pandas numpy")
    sys.exit(1)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Sabitler ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CACHE   = PROJECT_ROOT / "data_cache"
SANDBOX_DIR  = PROJECT_ROOT / "sandbox"

# SuperKAMA - Rust bot parametreleri (state.rs) → 1m için calibre edilmiş
# Diğer TF'ler için oran: 2584 bar @ 1m = ~43 saat
_KAMA_BASE_MINUTES = 2584   # 1m referansı
SUPERKAMA_FAST_PERIOD = 34
SUPERKAMA_SLOW_PERIOD = 55
SUPERKAMA_ATR_MULT    = 1.0    # Band genişliği

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}

def superkama_params(tf: str):
    """TF'e göre ölçeklenmiş KAMA uzunluğu ve ATR periyodu döner."""
    scale  = TF_MINUTES.get(tf, 15) / TF_MINUTES["1m"]
    length = max(50, round(_KAMA_BASE_MINUTES / scale))
    atr_p  = max(10, round(33 / scale))
    return length, atr_p


# ═══════════════════════════════════════════════════════════════════════════════
#  1. VERİ KATMANI
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}

BINANCE_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

HTF_MAP = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h"}


def download_binance(symbol: str, tf: str, days: int) -> pd.DataFrame:
    if not HAS_REQUESTS:
        raise ImportError("requests paketi gerekli: pip install requests")

    url   = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1500
    ms_tf = BINANCE_MS[tf]
    end   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86_400_000

    all_rows = []
    cur = start
    print(f"  Binance'tan indiriliyor: {symbol} {tf} ({days} gün)…")
    while cur < end:
        params = dict(symbol=symbol, interval=tf, startTime=cur,
                      endTime=end, limit=limit)
        data = requests.get(url, params=params, timeout=30).json()
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1][0] + ms_tf
        if len(data) < limit:
            break

    if not all_rows:
        raise ValueError(f"{symbol} {tf} için veri alınamadı")

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","trades","tbv","tqv","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✓ {len(df):,} mum indirildi  ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})")
    return df


def load_csv(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Cache klasöründen CSV yükler."""
    path = DATA_CACHE / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["open_time"])
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✓ {path.name}  →  {len(df):,} mum  ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})")
    return df


def get_data(symbol: str, tf: str, days: int, force_download: bool) -> pd.DataFrame:
    if not force_download:
        df = load_csv(symbol, tf)
        if df is not None:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            df = df[df["open_time"] >= cutoff].reset_index(drop=True)
            if len(df) > 100:
                return df
            print(f"  Cache yeterince veri içermiyor, Binance'tan çekiliyor…")

    return download_binance(symbol, tf, days)


def get_htf_data(symbol: str, tf: str, days: int, force_download: bool) -> Optional[pd.DataFrame]:
    htf = HTF_MAP.get(tf)
    if htf is None:
        return None
    print(f"  HTF ({htf}) verisi yükleniyor…")
    try:
        return get_data(symbol, htf, days + 5, force_download)
    except Exception as e:
        print(f"  ⚠ HTF veri alınamadı: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. İNDİKATÖRLER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    k = 2.0 / (period + 1)
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        if np.isnan(result[i-1]) if i > 0 else True:
            result[i] = series[i]
        else:
            result[i] = series[i] * k + result[i-1] * (1 - k)
    return result


def compute_kama(close: np.ndarray, length: int, fast: int, slow: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average - Rust Kama::update ile birebir."""
    n   = len(close)
    out = np.full(n, np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    # İlk geçerli değer = ilk kapanış
    out[0] = close[0]

    for i in range(1, n):
        # Efficiency Ratio: |close[i] - close[i-length]| / sum(|change|)
        if i >= length:
            direction  = abs(close[i] - close[i - length])
            volatility = np.sum(np.abs(np.diff(close[max(0, i-length):i+1])))
            er = direction / volatility if volatility > 0 else 0.0
        else:
            er = 0.0

        sc     = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev   = out[i-1] if not np.isnan(out[i-1]) else close[i]
        out[i] = prev + sc * (close[i] - prev)

    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int) -> np.ndarray:
    """Wilder ATR (1/period smoothing) - Rust Atr::update ile aynı."""
    n  = len(close)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl  = high[i] - low[i]
        hc  = abs(high[i] - close[i-1])
        lc  = abs(low[i]  - close[i-1])
        tr[i] = max(hl, hc, lc)

    atr = np.full(n, np.nan)
    alpha = 1.0 / period
    for i in range(n):
        if np.isnan(tr[i]):
            continue
        if np.isnan(atr[i-1]) if i > 0 else True:
            atr[i] = tr[i]
        else:
            atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    n  = len(close)
    adx = np.full(n, np.nan)
    atr = compute_atr(high, low, close, period)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up   = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i]  = up   if up > down and up > 0   else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

    def smooth(arr):
        s = np.full(n, np.nan)
        alpha = 1.0 / period
        for i in range(n):
            if np.isnan(s[i-1]) if i > 0 else True:
                s[i] = arr[i]
            else:
                s[i] = s[i-1] * (1 - alpha) + arr[i] * alpha
        return s

    s_plus  = smooth(plus_dm)
    s_minus = smooth(minus_dm)
    s_atr   = atr  # zaten Wilder

    di_plus  = 100 * s_plus  / np.where(s_atr > 0, s_atr, np.nan)
    di_minus = 100 * s_minus / np.where(s_atr > 0, s_atr, np.nan)
    di_sum   = di_plus + di_minus
    dx       = 100 * np.abs(di_plus - di_minus) / np.where(di_sum > 0, di_sum, np.nan)

    # ADX = smoothed DX
    adx_s = np.full(n, np.nan)
    alpha = 1.0 / period
    for i in range(n):
        if np.isnan(dx[i]):
            continue
        if np.isnan(adx_s[i-1]) if i > 0 else True:
            adx_s[i] = dx[i]
        else:
            adx_s[i] = adx_s[i-1] * (1 - alpha) + dx[i] * alpha
    return adx_s


def compute_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    vol_ma = pd.Series(volume).rolling(period, min_periods=1).mean().values
    return volume / np.where(vol_ma > 0, vol_ma, np.nan)


def compute_htf_trend(htf_df: pd.DataFrame, ltf_times: pd.Series,
                      htf_tf: str = "1h") -> np.ndarray:
    """
    HTF veriyi LTF zaman eksenine hizala: her LTF mumu için geçerli HTF trend
    döndür (1=yükselen, -1=düşen, 0=nötr).
    """
    htf_len, _ = superkama_params(htf_tf)
    htf_kama   = compute_kama(htf_df["close"].values,
                               htf_len, SUPERKAMA_FAST_PERIOD, SUPERKAMA_SLOW_PERIOD)
    htf_kama_rising  = np.diff(htf_kama, prepend=np.nan) > 0
    htf_close_kama   = htf_df["close"].values > htf_kama

    htf_trend = np.where(htf_kama_rising & htf_close_kama,  1,
                np.where(~htf_kama_rising & ~htf_close_kama, -1, 0))

    # Pandas merge_asof ile align
    htf_df2 = htf_df.copy()
    htf_df2["_htf_trend"] = htf_trend
    ltf_df2 = pd.DataFrame({"open_time": ltf_times})

    merged = pd.merge_asof(
        ltf_df2.sort_values("open_time"),
        htf_df2[["open_time","_htf_trend"]].sort_values("open_time"),
        on="open_time", direction="backward"
    )
    return merged["_htf_trend"].fillna(0).values


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SUPERKAMA SİNYAL MOTORU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_superkama_signals(df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None,
                              tf: str = "15m"):
    """
    Rust state.rs::update_pine_state'ın birebir Python karşılığı.
    Confluence puanlama ile zenginleştirilmiş.

    Dönen sütunlar:
      kama, kama_atr, upper_band, lower_band
      kama_rising, kama_falling
      buy_signal, sell_signal, strong_buy, strong_sell
      ema5, ema8, ema13, ema50, ema200
      adx, volume_ratio
      htf_trend  (HTF varsa)
      confluence_long, confluence_short  (0-6 puan)
    """
    kama_length, kama_atr_period = superkama_params(tf)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values

    # Temel indikatörler (TF'e göre ölçeklenmiş parametreler)
    kama     = compute_kama(c, kama_length, SUPERKAMA_FAST_PERIOD, SUPERKAMA_SLOW_PERIOD)
    kama_atr = compute_atr(h, l, c, kama_atr_period)
    ema5     = compute_ema(c, 5)
    ema8     = compute_ema(c, 8)
    ema13    = compute_ema(c, 13)
    ema50    = compute_ema(c, 50)
    ema200   = compute_ema(c, 200)
    adx      = compute_adx(h, l, c, 14)
    vol_r    = compute_volume_ratio(v, 20)

    n = len(c)

    upper_band = kama + kama_atr * SUPERKAMA_ATR_MULT
    lower_band = kama - kama_atr * SUPERKAMA_ATR_MULT

    # Kaydırımlı (önceki bar) değerler
    prev_close = np.roll(c, 1);      prev_close[0] = np.nan
    prev_kama  = np.roll(kama, 1);   prev_kama[0]  = np.nan
    prev_upper = np.roll(upper_band, 1); prev_upper[0] = np.nan
    prev_lower = np.roll(lower_band, 1); prev_lower[0] = np.nan

    kama_rising  = kama > prev_kama
    kama_falling = kama < prev_kama

    # Temel sinyaller (Rust parity)
    buy_signal   = (c > kama) & (prev_close <= prev_kama) & kama_rising
    sell_signal  = (c < kama) & (prev_close >= prev_kama) & kama_falling
    strong_buy   = (c > lower_band) & (prev_close <= prev_lower) & kama_rising
    strong_sell  = (c < upper_band) & (prev_close >= prev_upper) & kama_falling

    # HTF trend
    htf_tf = HTF_MAP.get(tf, "1h")
    if htf_df is not None:
        htf_trend = compute_htf_trend(htf_df, df["open_time"], htf_tf=htf_tf)
    else:
        htf_trend = np.zeros(n)

    # ── Confluence Puanı ────────────────────────────────────────────────────
    # LONG (0-6)
    # [1] EMA fan aligned: 5 > 8 > 13
    ema_fan_long  = (ema5 > ema8) & (ema8 > ema13)
    ema_fan_short = (ema5 < ema8) & (ema8 < ema13)
    # [2] EMA50 doğru tarafta
    ema50_long  = c > ema50
    ema50_short = c < ema50
    # [3] EMA200 doğru tarafta
    ema200_long  = c > ema200
    ema200_short = c < ema200
    # [4] ADX > 28 (trend gücü)
    adx_ok = adx > 28
    # [5] HTF aynı yön
    htf_long  = htf_trend == 1
    htf_short = htf_trend == -1
    # [6] Volume spike (> 1.3x)
    vol_ok = vol_r > 1.3

    conf_long  = (ema_fan_long.astype(int)  + ema50_long.astype(int)  +
                  ema200_long.astype(int)   + adx_ok.astype(int)      +
                  htf_long.astype(int)      + vol_ok.astype(int))

    conf_short = (ema_fan_short.astype(int) + ema50_short.astype(int) +
                  ema200_short.astype(int)  + adx_ok.astype(int)      +
                  htf_short.astype(int)     + vol_ok.astype(int))

    df2 = df.copy()
    df2["kama"]         = kama
    df2["kama_atr"]     = kama_atr
    df2["upper_band"]   = upper_band
    df2["lower_band"]   = lower_band
    df2["kama_rising"]  = kama_rising
    df2["kama_falling"] = kama_falling
    df2["buy_signal"]   = buy_signal
    df2["sell_signal"]  = sell_signal
    df2["strong_buy"]   = strong_buy
    df2["strong_sell"]  = strong_sell
    df2["ema5"]         = ema5
    df2["ema8"]         = ema8
    df2["ema13"]        = ema13
    df2["ema50"]        = ema50
    df2["ema200"]       = ema200
    df2["adx"]          = adx
    df2["volume_ratio"] = vol_r
    df2["htf_trend"]    = htf_trend
    df2["conf_long"]    = conf_long
    df2["conf_short"]   = conf_short
    return df2


# ═══════════════════════════════════════════════════════════════════════════════
#  4. BACKTEST MOTORU
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    min_confluence:    int   = 4       # Min puan (0-6)
    sl_atr_mult:       float = 2.0     # SL = ATR * mult
    tp_atr_mult:       float = 4.0     # TP = ATR * mult (0 = indicator_flip only)
    use_strong_signal: bool  = True    # strong_buy/sell de kabul et
    entry_confirm:     bool  = True    # Sonraki mumu bekle
    max_hold_candles:  int   = 200     # Maks tutma süresi
    risk_r:            float = 1.0     # Risk birimi (R)
    # Break Even mekanizması (Rust config'e paralel)
    be_threshold_candles: int   = 8    # N mum sonra kârdaysa SL=BE
    be_min_profit_r:      float = 0.0  # Minimum kâr (R) BE için


@dataclass
class Trade:
    direction:    int     # 1=LONG, -1=SHORT
    entry_price:  float
    entry_idx:    int
    sl:           float
    tp:           float
    atr:          float
    result:       str   = ""    # WIN | LOSS | BE | MAX_DUR
    exit_price:   float = 0.0
    exit_idx:     int   = 0
    pnl_r:        float = 0.0
    conf_score:   int   = 0
    is_strong:    bool  = False
    be_applied:   bool  = False   # SL break-even'e taşındı mı?


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> list[Trade]:
    trades: list[Trade] = []
    in_trade:  Optional[Trade] = None
    pending:   Optional[dict]  = None   # entry_confirm için bekleyen sinyal

    rows = df.to_dict("records")
    n    = len(rows)

    for i, row in enumerate(rows):
        if np.isnan(row["kama"]) or np.isnan(row["kama_atr"]):
            continue

        atr = row["kama_atr"]
        if atr <= 0:
            continue

        # ── Açık pozisyon yönetimi ─────────────────────────────────────────
        if in_trade is not None:
            hi = row["high"]
            lo = row["low"]
            cl = row["close"]
            t  = in_trade
            candles_held = i - t.entry_idx

            closed = False

            # ── Break Even mekanizması (Rust be_threshold_candles) ─────────
            if (not t.be_applied
                    and candles_held >= cfg.be_threshold_candles):
                if t.direction == 1:
                    unrealized_r = (cl - t.entry_price) / abs(t.entry_price - t.sl) * cfg.risk_r
                    if unrealized_r >= cfg.be_min_profit_r:
                        t.sl = t.entry_price   # SL → giriş fiyatına taşı
                        t.be_applied = True
                elif t.direction == -1:
                    unrealized_r = (t.entry_price - cl) / abs(t.sl - t.entry_price) * cfg.risk_r
                    if unrealized_r >= cfg.be_min_profit_r:
                        t.sl = t.entry_price
                        t.be_applied = True

            # TP kontrolü
            if cfg.tp_atr_mult > 0:
                if t.direction == 1 and hi >= t.tp:
                    t.exit_price = t.tp
                    t.exit_idx   = i
                    t.result     = "WIN"
                    t.pnl_r      = cfg.risk_r * (cfg.tp_atr_mult / cfg.sl_atr_mult)
                    closed = True
                elif t.direction == -1 and lo <= t.tp:
                    t.exit_price = t.tp
                    t.exit_idx   = i
                    t.result     = "WIN"
                    t.pnl_r      = cfg.risk_r * (cfg.tp_atr_mult / cfg.sl_atr_mult)
                    closed = True

            if not closed:
                # SL kontrolü
                if t.direction == 1 and lo <= t.sl:
                    t.exit_price = t.sl
                    t.exit_idx   = i
                    t.pnl_r      = (t.sl - t.entry_price) / abs(t.entry_price - t.atr * cfg.sl_atr_mult + t.entry_price - t.sl + 0.0001) * cfg.risk_r if not t.be_applied else 0.0
                    t.result     = "BE" if t.be_applied else "LOSS"
                    if not t.be_applied:
                        t.pnl_r = -cfg.risk_r
                    closed = True
                elif t.direction == -1 and hi >= t.sl:
                    t.exit_price = t.sl
                    t.exit_idx   = i
                    t.result     = "BE" if t.be_applied else "LOSS"
                    t.pnl_r      = 0.0 if t.be_applied else -cfg.risk_r
                    closed = True

            if not closed:
                # Indicator flip (karşı sinyal)
                if t.direction == 1 and _is_sell(row, cfg):
                    pnl_r  = (cl - t.entry_price) / abs(t.entry_price - t.sl) * cfg.risk_r
                    t.exit_price = cl
                    t.exit_idx   = i
                    t.pnl_r      = pnl_r
                    t.result     = "WIN" if pnl_r > 0.05 else ("BE" if pnl_r >= -0.05 else "LOSS")
                    closed = True
                elif t.direction == -1 and _is_buy(row, cfg):
                    pnl_r  = (t.entry_price - cl) / abs(t.sl - t.entry_price) * cfg.risk_r
                    t.exit_price = cl
                    t.exit_idx   = i
                    t.pnl_r      = pnl_r
                    t.result     = "WIN" if pnl_r > 0.05 else ("BE" if pnl_r >= -0.05 else "LOSS")
                    closed = True

            if not closed:
                # Max dur
                if (i - t.entry_idx) >= cfg.max_hold_candles:
                    pnl_r  = ((cl - t.entry_price) / (abs(t.entry_price - t.sl))
                              * t.direction * cfg.risk_r)
                    t.exit_price = cl
                    t.exit_idx   = i
                    t.pnl_r      = pnl_r
                    t.result     = "MAX_DUR"
                    closed = True

            if closed:
                trades.append(in_trade)
                in_trade = None

        # ── Bekleyen sinyal onayı ──────────────────────────────────────────
        if in_trade is None and pending is not None:
            # Bir önceki mumun sinyalini bu mumda onaylarız
            prev_row   = rows[pending["idx"]]
            direction  = pending["direction"]
            conf_score = pending["conf_score"]
            is_strong  = pending["is_strong"]
            entry_atr  = prev_row["kama_atr"]

            # Giriş fiyatı: açılış (bir sonraki mum)
            entry = row["open"]
            chase_limit = entry_atr * 0.3   # max %0.3 ATR uzaklaşma

            if direction == 1:
                if row["open"] > prev_row["close"] * 1.005:
                    pending = None
                    continue  # Hızlı yukarı gap = kaçtı, geç
                sl  = entry - entry_atr * cfg.sl_atr_mult
                tp  = entry + entry_atr * cfg.tp_atr_mult if cfg.tp_atr_mult > 0 else 0.0
            else:
                if row["open"] < prev_row["close"] * 0.995:
                    pending = None
                    continue
                sl  = entry + entry_atr * cfg.sl_atr_mult
                tp  = entry - entry_atr * cfg.tp_atr_mult if cfg.tp_atr_mult > 0 else 0.0

            in_trade = Trade(
                direction   = direction,
                entry_price = entry,
                entry_idx   = i,
                sl          = sl,
                tp          = tp,
                atr         = entry_atr,
                conf_score  = conf_score,
                is_strong   = is_strong,
            )
            pending = None

        # ── Yeni sinyal arama ──────────────────────────────────────────────
        if in_trade is None and pending is None:
            direction   = 0
            conf_score  = 0
            is_strong   = False

            if _is_buy(row, cfg):
                direction  = 1
                conf_score = row["conf_long"]
                is_strong  = bool(row["strong_buy"])
            elif _is_sell(row, cfg):
                direction  = -1
                conf_score = row["conf_short"]
                is_strong  = bool(row["strong_sell"])

            if direction != 0 and conf_score >= cfg.min_confluence:
                if cfg.entry_confirm:
                    pending = dict(idx=i, direction=direction,
                                   conf_score=conf_score, is_strong=is_strong)
                else:
                    entry  = row["close"]
                    atr_e  = row["kama_atr"]
                    if direction == 1:
                        sl = entry - atr_e * cfg.sl_atr_mult
                        tp = entry + atr_e * cfg.tp_atr_mult if cfg.tp_atr_mult > 0 else 0.0
                    else:
                        sl = entry + atr_e * cfg.sl_atr_mult
                        tp = entry - atr_e * cfg.tp_atr_mult if cfg.tp_atr_mult > 0 else 0.0
                    in_trade = Trade(
                        direction   = direction,
                        entry_price = entry,
                        entry_idx   = i,
                        sl          = sl,
                        tp          = tp,
                        atr         = atr_e,
                        conf_score  = conf_score,
                        is_strong   = is_strong,
                    )

    return trades


def _is_buy(row: dict, cfg: BacktestConfig) -> bool:
    return bool(row["buy_signal"]) or (cfg.use_strong_signal and bool(row["strong_buy"]))


def _is_sell(row: dict, cfg: BacktestConfig) -> bool:
    return bool(row["sell_signal"]) or (cfg.use_strong_signal and bool(row["strong_sell"]))


# ═══════════════════════════════════════════════════════════════════════════════
#  5. METRİKLER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    total_trades:   int   = 0    # Sadece WIN + LOSS (BE hariç, Rust'a paralel)
    total_opened:   int   = 0    # Tüm açılan işlemler
    be_count:       int   = 0    # Break Even çıkışları
    wins:           int   = 0
    losses:         int   = 0
    win_rate:       float = 0.0
    pnl_r:          float = 0.0
    avg_win_r:      float = 0.0
    avg_loss_r:     float = 0.0
    expectancy:     float = 0.0
    profit_factor:  float = 0.0
    sharpe:         float = 0.0
    max_drawdown_r: float = 0.0
    max_consec_loss: int  = 0
    trades_per_day: float = 0.0
    cfg:            Optional[BacktestConfig] = None


def compute_metrics(trades: list[Trade], days: int,
                    cfg: BacktestConfig) -> BacktestResult:
    if not trades:
        return BacktestResult(cfg=cfg)

    total_opened = len(trades)
    be_trades    = [t for t in trades if t.result == "BE"]
    decisive     = [t for t in trades if t.result in ("WIN", "LOSS")]

    if not decisive:
        return BacktestResult(total_opened=total_opened,
                              be_count=len(be_trades), cfg=cfg)

    wins   = [t for t in decisive if t.result == "WIN"]
    losses = [t for t in decisive if t.result == "LOSS"]

    total  = len(decisive)
    n_win  = len(wins)
    n_loss = len(losses)
    wr     = n_win / total

    avg_win  = np.mean([t.pnl_r for t in wins])   if wins   else 0.0
    avg_loss = np.mean([t.pnl_r for t in losses]) if losses else 0.0
    gross_w  = sum(t.pnl_r for t in wins)
    gross_l  = abs(sum(t.pnl_r for t in losses))

    pnl_r        = sum(t.pnl_r for t in trades)
    expectancy   = wr * avg_win + (1-wr) * avg_loss
    profit_factor = gross_w / gross_l if gross_l > 0 else float("inf")

    # Sharpe (tüm işlemler üzerinden)
    pnls = np.array([t.pnl_r for t in decisive])
    sharpe = (np.mean(pnls) / np.std(pnls) * math.sqrt(252)) if np.std(pnls) > 0 else 0.0

    # Max drawdown
    equity = np.cumsum([t.pnl_r for t in trades])
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0

    # Max consecutive losses
    best_run = cur_run = 0
    for t in decisive:
        if t.result == "LOSS":
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 0

    tpd = total / max(days, 1)

    return BacktestResult(
        total_trades    = total,
        total_opened    = total_opened,
        be_count        = len(be_trades),
        wins            = n_win,
        losses          = n_loss,
        win_rate        = wr * 100,
        pnl_r           = pnl_r,
        avg_win_r       = avg_win,
        avg_loss_r      = avg_loss,
        expectancy      = expectancy,
        profit_factor   = profit_factor,
        sharpe          = sharpe,
        max_drawdown_r  = max_dd,
        max_consec_loss = best_run,
        trades_per_day  = tpd,
        cfg             = cfg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search(df: pd.DataFrame, days: int) -> list[BacktestResult]:
    print("\n" + "═"*60)
    print("  GRID SEARCH başlatılıyor…")
    print("═"*60)

    param_grid = {
        "min_confluence":       [2, 3, 4, 5],
        "sl_atr_mult":          [1.5, 2.0, 2.5],
        "tp_atr_mult":          [3.0, 4.0, 5.0],
        "be_threshold_candles": [0, 5, 8, 15],   # 0 = BE yok
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    results = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg    = BacktestConfig(**params)
        trades = run_backtest(df, cfg)
        res    = compute_metrics(trades, days, cfg)

        if idx % 12 == 0:
            print(f"  [{idx+1:3d}/{total}] conf={params['min_confluence']}  "
                  f"sl={params['sl_atr_mult']}×  tp={params['tp_atr_mult']}×  "
                  f"be={params['be_threshold_candles']}c  "
                  f"→  {res.total_trades}({res.be_count}BE) işlem  %{res.win_rate:.1f} WR  "
                  f"{res.pnl_r:+.2f}R  EV={res.expectancy:+.3f}")
        results.append(res)

    return results


def best_by(results: list[BacktestResult], key: str, min_trades: int = 10):
    filtered = [r for r in results if r.total_trades >= min_trades]
    if not filtered:
        return None
    return max(filtered, key=lambda r: getattr(r, key))


# ═══════════════════════════════════════════════════════════════════════════════
#  7. RAPOR
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(label: str, res: BacktestResult):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if res.cfg:
        c = res.cfg
        print(f"  Config  →  conf={c.min_confluence}  SL={c.sl_atr_mult}×ATR  "
              f"TP={'flip' if c.tp_atr_mult==0 else str(c.tp_atr_mult)+'×ATR'}  "
              f"BE_after={c.be_threshold_candles}  confirm={c.entry_confirm}")
    be_pct = res.be_count / res.total_opened * 100 if res.total_opened > 0 else 0
    print(f"  Açılan işlem    : {res.total_opened}  "
          f"(D: WIN:{res.wins}/LOSS:{res.losses}  |  BE:{res.be_count} = %{be_pct:.1f})")
    print(f"  Win Rate        : {res.win_rate:.2f}%")
    print(f"  PNL (R)         : {res.pnl_r:+.4f}R")
    print(f"  Ortalama Kazanç : {res.avg_win_r:+.3f}R")
    print(f"  Ortalama Kayıp  : {res.avg_loss_r:+.3f}R")
    print(f"  Beklenti (EV)   : {res.expectancy:+.4f}R/işlem")
    print(f"  Profit Factor   : {res.profit_factor:.3f}")
    print(f"  Sharpe Ratio    : {res.sharpe:.3f}")
    print(f"  Max DD (R)      : -{res.max_drawdown_r:.3f}R")
    print(f"  Max Consec Loss : {res.max_consec_loss}")
    print(f"  İşlem/gün       : {res.trades_per_day:.3f}")


def save_results(results: list[BacktestResult], symbol: str, tf: str):
    out = []
    for r in results:
        d = {k: v for k, v in vars(r).items() if k != "cfg"}
        if r.cfg:
            d.update({
                "min_confluence":       r.cfg.min_confluence,
                "sl_atr_mult":          r.cfg.sl_atr_mult,
                "tp_atr_mult":          r.cfg.tp_atr_mult,
                "be_threshold_candles": r.cfg.be_threshold_candles,
            })
        out.append(d)

    path = SANDBOX_DIR / f"grid_{symbol}_{tf}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Grid sonuçları → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Precision Confluence Backtest Sandbox")
    p.add_argument("--symbol",    default="ETHUSDT")
    p.add_argument("--tf",        default="15m")
    p.add_argument("--days",      type=int,   default=365)
    p.add_argument("--download",  action="store_true")
    p.add_argument("--grid",      action="store_true", default=True)
    p.add_argument("--no_grid",   action="store_true")
    p.add_argument("--min_score", type=int,   default=None)
    p.add_argument("--sl_atr",    type=float, default=2.0)
    p.add_argument("--tp_atr",    type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    tf     = args.tf
    days   = args.days
    run_grid = not args.no_grid

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Precision Confluence Backtest Sandbox             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Sembol: {symbol}  |  TF: {tf}  |  Süre: {days} gün")

    # Veri yükle
    print("\n[1] Veri yükleniyor…")
    df     = get_data(symbol, tf, days, args.download)
    htf_df = get_htf_data(symbol, tf, days, args.download)

    # İndikatörler + sinyaller
    print("\n[2] İndikatörler hesaplanıyor…")
    kl, ka = superkama_params(tf)
    print(f"  KAMA params → length={kl}, ATR_period={ka}  (TF={tf})")
    df = compute_superkama_signals(df, htf_df, tf=tf)

    total_signals = (df["buy_signal"] | df["sell_signal"] |
                     df["strong_buy"] | df["strong_sell"]).sum()
    print(f"  Ham sinyal sayısı (filtresiz): {total_signals:,}")
    print(f"  ADX > 28 olan mumlar: {(df['adx'] > 28).sum():,} / {len(df):,}")
    print(f"  Volume spike mumlar: {(df['volume_ratio'] > 1.3).sum():,} / {len(df):,}")
    htf_long_pct  = (df["htf_trend"] == 1).mean() * 100
    htf_short_pct = (df["htf_trend"] == -1).mean() * 100
    print(f"  HTF trend  LONG: %{htf_long_pct:.1f}  SHORT: %{htf_short_pct:.1f}")

    # Grid Search
    if run_grid and args.min_score is None:
        print("\n[3] Grid Search…")
        all_results = grid_search(df, days)
        save_results(all_results, symbol, tf)

        print("\n" + "═"*60)
        print("  EN İYİ SONUÇLAR")
        print("═"*60)

        best_wr      = best_by(all_results, "win_rate", min_trades=15)
        best_sharpe  = best_by(all_results, "sharpe",   min_trades=15)
        best_pnl     = best_by(all_results, "pnl_r",    min_trades=15)
        best_ev      = best_by(all_results, "expectancy", min_trades=15)

        if best_wr:     print_result("► En yüksek Win Rate", best_wr)
        if best_sharpe: print_result("► En yüksek Sharpe", best_sharpe)
        if best_pnl:    print_result("► En yüksek PNL (R)", best_pnl)
        if best_ev:     print_result("► En yüksek Beklenti (EV)", best_ev)

    else:
        # Tek koşum
        min_sc = args.min_score if args.min_score is not None else 4
        cfg    = BacktestConfig(
            min_confluence = min_sc,
            sl_atr_mult    = args.sl_atr,
            tp_atr_mult    = args.tp_atr,
        )
        print(f"\n[3] Backtest  (conf ≥ {min_sc},  SL={args.sl_atr}×ATR,  TP={args.tp_atr}×ATR)…")
        trades = run_backtest(df, cfg)
        res    = compute_metrics(trades, days, cfg)
        print_result(f"{symbol} {tf} – Precision Backtest", res)

    print("\n  ✓ Tamamlandı.\n")


if __name__ == "__main__":
    main()
