"""
==============================================================================
 EXPERIMENT LAB  — Kapsamlı 1m İndikatör & SL/TP Optimizasyon Sistemi
==============================================================================
Hedef:
  • Her indikatör sistemini AYRI AYRI test et (SuperKAMA, EMA cross, MACD,
    RSI, Bollinger, Supertrend, Stoch, ADX+DI, Pivot, VWAP-slope, OBV-slope)
  • SL/TP'yi hardcode DEĞİL, matematiksel olarak hesapla:
      – ATR percentile (dinamik)
      – Tarihsel candle range dağılımı
      – Kelly Criterion (optimal R çarpanı)
      – Optimal-F (Ralph Vince)
      – Risk-of-Ruin eşiği
  • Walk-Forward Validation (overfitting'i önle)
  • Monte Carlo (n=500) ile güven aralığı
  • Her zaman 1m üzerinde çalış

Kullanım:
  python sandbox/experiment_lab.py --symbol ETHUSDT --days 365
  python sandbox/experiment_lab.py --symbol BTCUSDT --days 180 --systems superkama,ema,macd
  python sandbox/experiment_lab.py --download --symbol SOLUSDT
  python sandbox/experiment_lab.py --summary          # En iyi sonuçları göster
==============================================================================
"""

import argparse, json, math, os, sys, warnings, itertools, hashlib, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pandas ve numpy gerekli: pip install pandas numpy")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─── Yollar ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data_cache"
LAB_DIR   = ROOT / "sandbox" / "lab_results"
LAB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

TF = "1m"   # ASLA DEĞİŞMEZ


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 1 — VERİ KATMANI
# ══════════════════════════════════════════════════════════════════════════════

BINANCE_BASE  = "https://fapi.binance.com"
TF_MS         = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
                 "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}


def download_klines(symbol: str, tf: str, days: int,
                    progress: bool = True) -> pd.DataFrame:
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")
    ms   = TF_MS[tf]
    end  = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86_400_000
    rows = []
    cur  = start
    limit = 1500
    total_expected = (days * 86_400_000) // ms
    fetched = 0
    if progress:
        print(f"  >> {symbol} {tf} {days}g - yaklasik {total_expected:,} mum...", end="", flush=True)
    while cur < end:
        resp = requests.get(f"{BINANCE_BASE}/fapi/v1/klines",
                            params=dict(symbol=symbol, interval=tf,
                                        startTime=cur, endTime=end,
                                        limit=limit),
                            timeout=30).json()
        if not resp:
            break
        rows.extend(resp)
        cur = resp[-1][0] + ms
        fetched += len(resp)
        if progress and fetched % 10000 < limit:
            print(f"\r  >> {symbol} {tf}  {fetched:>7,}/{total_expected:,}", end="", flush=True)
        if len(resp) < limit:
            break
    if progress:
        print(f"\r  OK {symbol} {tf}  {len(rows):,} mum indirildi" + " "*20)
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qv","trades","tbv","tqv","ignore"]
    df = pd.DataFrame(rows, columns=cols)[["open_time","open","high","low","close","volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_or_download(symbol: str, days: int, force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_1m.csv"
    if not force and path.exists():
        df = pd.read_csv(path, parse_dates=["open_time"])
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize("UTC")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["open_time"] >= cutoff].reset_index(drop=True)
        if len(df) >= days * 1200:   # en az başlangıç verisi
            print(f"  OK Cache: {path.name} -> {len(df):,} mum")
            return df
        print(f"  !! Cache yetersiz ({len(df):,}), yeniden indiriliyor...")

    df = download_klines(symbol, "1m", days + 5)
    df.to_csv(path, index=False)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df["open_time"] >= cutoff].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 2 — HIZLI İNDİKATÖR KÜTÜPHANESİ (vectorized numpy)
# ══════════════════════════════════════════════════════════════════════════════

def ema(s: np.ndarray, p: int) -> np.ndarray:
    k, out = 2/(p+1), np.full(len(s), np.nan)
    for i in range(len(s)):
        if np.isnan(s[i]): continue
        out[i] = s[i] if (i == 0 or np.isnan(out[i-1])) else s[i]*k + out[i-1]*(1-k)
    return out

def wilder_smooth(s: np.ndarray, p: int) -> np.ndarray:
    """Wilder's 1/p smoothing (ATR, ADX)."""
    out = np.full(len(s), np.nan)
    alpha = 1/p
    for i in range(len(s)):
        if np.isnan(s[i]): continue
        out[i] = s[i] if (i == 0 or np.isnan(out[i-1])) else out[i-1]*(1-alpha)+s[i]*alpha
    return out

def atr(h, l, c, p=14) -> np.ndarray:
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    tr[0] = h[0]-l[0]
    return wilder_smooth(tr, p)

def rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=np.nan)
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    avg_u = wilder_smooth(up, p)
    avg_d = wilder_smooth(dn, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_d == 0, 100.0, avg_u / avg_d)
    return 100 - 100/(1+rs)

def macd(c, fast=12, slow=26, sig=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return m, s, m - s

def bollinger(c, p=20, std_mult=2.0):
    mid = pd.Series(c).rolling(p, min_periods=1).mean().values
    std = pd.Series(c).rolling(p, min_periods=1).std().values
    return mid - std_mult*std, mid, mid + std_mult*std

def kama(c, length=10, fast=2, slow=30) -> np.ndarray:
    fc = 2/(fast+1); sc = 2/(slow+1)
    out = np.full(len(c), np.nan); out[0] = c[0]
    for i in range(1, len(c)):
        if i >= length:
            direction  = abs(c[i] - c[i-length])
            volatility = np.sum(np.abs(np.diff(c[max(0,i-length):i+1])))
            er = direction/volatility if volatility > 0 else 0
        else:
            er = 0
        sc2 = (er*(fc-sc)+sc)**2
        prev = out[i-1] if not np.isnan(out[i-1]) else c[i]
        out[i] = prev + sc2*(c[i]-prev)
    return out

def adx(h, l, c, p=14):
    up   = h - np.roll(h, 1); up[0] = 0
    down = np.roll(l, 1) - l; down[0] = 0
    pdm  = np.where((up > down) & (up > 0), up, 0)
    ndm  = np.where((down > up) & (down > 0), down, 0)
    _atr = atr(h, l, c, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = 100 * wilder_smooth(pdm, p) / np.where(_atr>0, _atr, np.nan)
        ndi = 100 * wilder_smooth(ndm, p) / np.where(_atr>0, _atr, np.nan)
        dxv = 100 * np.abs(pdi-ndi) / np.where((pdi+ndi)>0, pdi+ndi, np.nan)
    return wilder_smooth(dxv, p), pdi, ndi

def stoch_rsi(c, rsi_p=14, stoch_p=14, sk=3, sd=3):
    r = rsi(c, rsi_p)
    r_high = pd.Series(r).rolling(stoch_p, min_periods=1).max().values
    r_low  = pd.Series(r).rolling(stoch_p, min_periods=1).min().values
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_k = 100*(r - r_low)/np.where((r_high-r_low)>0, r_high-r_low, np.nan)
    k_ = ema(np.nan_to_num(raw_k, nan=50), sk)
    d_ = ema(k_, sd)
    return k_, d_

def supertrend(h, l, c, p=10, mult=3.0):
    a = atr(h, l, c, p)
    hl2 = (h+l)/2
    up_band  = hl2 - mult*a
    dn_band  = hl2 + mult*a
    trend    = np.ones(len(c))
    final_up = up_band.copy()
    final_dn = dn_band.copy()
    for i in range(1, len(c)):
        final_up[i] = max(up_band[i], final_up[i-1]) if c[i-1] > final_up[i-1] else up_band[i]
        final_dn[i] = min(dn_band[i], final_dn[i-1]) if c[i-1] < final_dn[i-1] else dn_band[i]
        trend[i] = (1 if c[i] > final_dn[i-1] else
                   -1 if c[i] < final_up[i-1] else trend[i-1])
    return trend, final_up, final_dn

def vwap_slope(c, v, p=20) -> np.ndarray:
    """VWAP'ın p-bar eğimi (normalize)."""
    pv   = pd.Series(c*v).rolling(p, min_periods=1).sum().values
    vol  = pd.Series(v).rolling(p, min_periods=1).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vw = pv / np.where(vol>0, vol, np.nan)
    slope = vw - np.roll(vw, p)
    slope[:p] = np.nan
    return slope

def obv_slope(c, v, p=20) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    obv_  = np.cumsum(np.where(delta > 0, v, np.where(delta < 0, -v, 0)))
    obv_ma = pd.Series(obv_).rolling(5, min_periods=1).mean().values
    slope  = obv_ma - np.roll(obv_ma, p)
    slope[:p] = np.nan
    return slope

def pivot_signals(h, l, c, lookback=5) -> tuple:
    """Pivot kırılma sinyalleri."""
    n = len(c)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    for i in range(lookback*2, n):
        ph_range = h[i-lookback*2:i-lookback]
        pl_range = l[i-lookback*2:i-lookback]
        if len(ph_range) == 0:
            continue
        ph = ph_range.max()
        pl = pl_range.min()
        if c[i] > ph and c[i-1] <= ph:
            long_sig[i] = True
        if c[i] < pl and c[i-1] >= pl:
            short_sig[i] = True
    return long_sig, short_sig


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 3 — SİNYAL SİSTEMLERİ KATALOĞU
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalDef:
    name:        str
    params:      dict
    generate:    Callable   = field(repr=False)

    
def _make_systems(df: pd.DataFrame) -> list[SignalDef]:
    """
    Tüm sinyal sistemlerini oluştur.
    Her sistem (long_arr, short_arr) döner — boolean numpy arrays.
    """
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)

    systems: list[SignalDef] = []

    # ── 1. SuperKAMA (Rust bot parity) ──
    for klen in [100, 200, 500]:
        for atr_mult in [0.8, 1.0, 1.2]:
            def _sk(h=h, l=l, c=c, klen=klen, atr_mult=atr_mult):
                k   = kama(c, klen, 34, 55)
                ka  = atr(h, l, c, 14) * atr_mult
                pk  = np.roll(k, 1); pk[0] = np.nan
                pc  = np.roll(c, 1); pc[0] = np.nan
                kr  = k > pk
                kf  = k < pk
                lo  = (c > k) & (pc <= pk) & kr
                sh  = (c < k) & (pc >= pk) & kf
                bnd_lo = k - ka; bnd_hi = k + ka
                pbh = np.roll(bnd_hi, 1); pbl = np.roll(bnd_lo, 1)
                pbh[0] = pbl[0] = np.nan
                slo = (c > bnd_lo) & (pc <= pbl) & kr
                ssh = (c < bnd_hi) & (pc >= pbh) & kf
                return lo | slo, sh | ssh
            systems.append(SignalDef(
                f"superkama_k{klen}_m{atr_mult}", {"klen": klen, "atr_mult": atr_mult},
                _sk,
            ))

    # ── 2. EMA Crossover (çoklu periyot) ──
    for fast, slow in [(5,13),(5,21),(8,21),(9,21),(13,34),(21,55),(34,89)]:
        def _emacross(c=c, fast=fast, slow=slow):
            ef = ema(c, fast); es = ema(c, slow)
            pef = np.roll(ef,1); pes = np.roll(es,1)
            lo = (ef > es) & (pef <= pes)
            sh = (ef < es) & (pef >= pes)
            return lo, sh
        systems.append(SignalDef(
            f"emacross_{fast}_{slow}", {"fast": fast, "slow": slow}, _emacross))

    # ── 3. EMA Triple Fan (5/8/13 gibi yapılar) ──
    for f,m,s in [(5,8,13),(8,13,21),(13,21,34),(21,34,55)]:
        def _emafan(c=c, f=f, m=m, s=s):
            ef = ema(c,f); em = ema(c,m); es = ema(c,s)
            pf = np.roll(ef,1); pm = np.roll(em,1); ps = np.roll(es,1)
            lo = (ef > em) & (em > es) & ~((pf > pm) & (pm > ps))
            sh = (ef < em) & (em < es) & ~((pf < pm) & (pm < ps))
            return lo, sh
        systems.append(SignalDef(
            f"emafan_{f}_{m}_{s}", {"f":f,"m":m,"s":s}, _emafan))

    # ── 4. MACD Crossover ──
    for fast, slow, sig in [(12,26,9),(8,21,5),(5,13,3),(3,10,3)]:
        def _macd(c=c, fast=fast, slow=slow, sig=sig):
            ml, sl_, hist = macd(c, fast, slow, sig)
            pml = np.roll(ml,1); psl = np.roll(sl_,1)
            lo  = (ml > sl_) & (pml <= psl)
            sh  = (ml < sl_) & (pml >= psl)
            return lo, sh
        systems.append(SignalDef(
            f"macd_{fast}_{slow}_{sig}", {"fast":fast,"slow":slow,"sig":sig}, _macd))

    # ── 5. MACD Histogram Direction Change ──
    for fast, slow, sig in [(12,26,9),(8,21,5)]:
        def _macd_hist(c=c, fast=fast, slow=slow, sig=sig):
            _, _, hist = macd(c, fast, slow, sig)
            ph = np.roll(hist,1)
            lo = (hist > 0) & (ph <= 0)
            sh = (hist < 0) & (ph >= 0)
            return lo, sh
        systems.append(SignalDef(
            f"macd_hist_{fast}_{slow}_{sig}", {}, _macd_hist))

    # ── 6. RSI Reversal (overbought/oversold) ──
    for p, ob, os in [(14,70,30),(14,65,35),(7,75,25),(21,70,30)]:
        def _rsi_rev(c=c, p=p, ob=ob, os=os):
            r  = rsi(c, p); pr = np.roll(r,1)
            lo = (r > os) & (pr <= os)
            sh = (r < ob) & (pr >= ob)
            return lo, sh
        systems.append(SignalDef(
            f"rsi_{p}_ob{ob}_os{os}", {"p":p,"ob":ob,"os":os}, _rsi_rev))

    # ── 7. RSI Midline Cross ──
    for p in [7, 14, 21]:
        def _rsi_mid(c=c, p=p):
            r  = rsi(c, p); pr = np.roll(r,1)
            lo = (r > 50) & (pr <= 50)
            sh = (r < 50) & (pr >= 50)
            return lo, sh
        systems.append(SignalDef(f"rsi_mid_{p}", {"p":p}, _rsi_mid))

    # ── 8. Bollinger Squeeze + Breakout ──
    for p, sd_m in [(20,2.0),(20,1.5),(10,2.0),(30,2.0)]:
        def _bb_break(h=h, l=l, c=c, p=p, sd_m=sd_m):
            bl, bm, bu = bollinger(c, p, sd_m)
            pc = np.roll(c,1)
            lo = (c > bu) & (pc <= bu)
            sh = (c < bl) & (pc >= bl)
            return lo, sh
        systems.append(SignalDef(
            f"bb_break_{p}_sd{sd_m}", {"p":p,"sd":sd_m}, _bb_break))

    # ── 9. Bollinger Mean Reversion ──
    for p, sd_m in [(20,2.0),(20,1.5)]:
        def _bb_rev(c=c, p=p, sd_m=sd_m):
            bl, bm, bu = bollinger(c, p, sd_m)
            pc = np.roll(c,1)
            lo = (c > bl) & (pc <= bl)   # alt band'dan yukarı kırma
            sh = (c < bu) & (pc >= bu)
            return lo, sh
        systems.append(SignalDef(
            f"bb_rev_{p}_sd{sd_m}", {"p":p,"sd":sd_m}, _bb_rev))

    # ── 10. Stochastic RSI cross ──
    for rp, sp, ob, os in [(14,14,80,20),(10,10,80,20),(14,14,70,30)]:
        def _stochrsi(c=c, rp=rp, sp=sp, ob=ob, os=os):
            k_, d_ = stoch_rsi(c, rp, sp)
            pk = np.roll(k_,1); pd2 = np.roll(d_,1)
            lo = (k_ > d_) & (pk <= pd2) & (k_ < ob)
            sh = (k_ < d_) & (pk >= pd2) & (k_ > os)
            return lo, sh
        systems.append(SignalDef(
            f"stochrsi_{rp}_{sp}_ob{ob}", {"rp":rp,"sp":sp}, _stochrsi))

    # ── 11. ADX+DI Crossover ──
    for p, adx_min in [(14,20),(14,25),(10,20)]:
        def _adxdi(h=h, l=l, c=c, p=p, adx_min=adx_min):
            adx_, pdi, ndi = adx(h, l, c, p)
            pp = np.roll(pdi,1); np2 = np.roll(ndi,1)
            strong = adx_ > adx_min
            lo = (pdi > ndi) & (pp <= np2) & strong
            sh = (pdi < ndi) & (pp >= np2) & strong
            return lo, sh
        systems.append(SignalDef(
            f"adxdi_{p}_min{adx_min}", {"p":p,"adx_min":adx_min}, _adxdi))

    # ── 12. Supertrend ──
    for p, mult in [(10,2.0),(10,3.0),(7,3.0),(14,2.0),(20,3.0)]:
        def _st(h=h, l=l, c=c, p=p, mult=mult):
            tr, _, _ = supertrend(h, l, c, p, mult)
            ptr = np.roll(tr, 1)
            lo  = (tr == 1) & (ptr == -1)
            sh  = (tr == -1) & (ptr == 1)
            return lo, sh
        systems.append(SignalDef(
            f"supertrend_{p}_m{mult}", {"p":p,"mult":mult}, _st))

    # ── 13. VWAP Slope ──
    for p in [20, 50, 100]:
        def _vwap(c=c, v=v, p=p):
            sl = vwap_slope(c, v, p)
            ps = np.roll(sl,1)
            lo = (sl > 0) & (ps <= 0)
            sh = (sl < 0) & (ps >= 0)
            return lo, sh
        systems.append(SignalDef(f"vwap_slope_{p}", {"p":p}, _vwap))

    # ── 14. OBV Slope ──
    for p in [20, 50]:
        def _obv(c=c, v=v, p=p):
            sl = obv_slope(c, v, p)
            ps = np.roll(sl,1)
            lo = (sl > 0) & (ps <= 0)
            sh = (sl < 0) & (ps >= 0)
            return lo, sh
        systems.append(SignalDef(f"obv_slope_{p}", {"p":p}, _obv))

    # ── 15. Pivot Kırılma ──
    for lb in [5, 10, 20]:
        def _piv(h=h, l=l, c=c, lb=lb):
            return pivot_signals(h, l, c, lb)
        systems.append(SignalDef(f"pivot_break_{lb}", {"lb":lb}, _piv))

    # ── 16. EMA + RSI Combo ──
    for ema_p, rsi_p, rsi_mid in [(21, 14, 50), (50, 14, 50), (200, 14, 50)]:
        def _ema_rsi(c=c, ep=ema_p, rp=rsi_p, rm=rsi_mid):
            e = ema(c, ep); pe = np.roll(e,1)
            r = rsi(c, rp)
            lo = (c > e) & (np.roll(c,1) <= pe) & (r > rm)
            sh = (c < e) & (np.roll(c,1) >= pe) & (r < rm)
            return lo, sh
        systems.append(SignalDef(
            f"ema{ema_p}_rsi{rsi_p}", {"ep":ema_p,"rp":rsi_p}, _ema_rsi))

    # ── 17. EMA + ADX Combo ──
    for ema_p, adx_min in [(21,20),(50,25),(200,20)]:
        def _ema_adx(h=h, l=l, c=c, ep=ema_p, am=adx_min):
            e  = ema(c, ep); pe = np.roll(e,1)
            adx_, pdi, ndi = adx(h, l, c, 14)
            lo = (c > e) & (np.roll(c,1) <= pe) & (adx_ > am) & (pdi > ndi)
            sh = (c < e) & (np.roll(c,1) >= pe) & (adx_ > am) & (ndi > pdi)
            return lo, sh
        systems.append(SignalDef(
            f"ema{ema_p}_adx{adx_min}", {"ep":ema_p,"am":adx_min}, _ema_adx))

    # ── 18. Supertrend + RSI ──
    for st_p, st_m, rsi_p in [(10,3.0,14),(7,3.0,14)]:
        def _st_rsi(h=h, l=l, c=c, sp=st_p, sm=st_m, rp=rsi_p):
            tr, _, _ = supertrend(h, l, c, sp, sm)
            ptr  = np.roll(tr, 1)
            r    = rsi(c, rp)
            lo   = (tr == 1) & (ptr == -1) & (r > 40)
            sh   = (tr == -1) & (ptr == 1) & (r < 60)
            return lo, sh
        systems.append(SignalDef(
            f"st{st_p}m{st_m}_rsi{rsi_p}", {}, _st_rsi))

    # ── 19. KAMA + MACD Combo ──
    for klen, macd_fast, macd_slow in [(100,8,21),(200,12,26)]:
        def _kama_macd(c=c, kl=klen, mf=macd_fast, ms=macd_slow):
            k   = kama(c, kl, 34, 55)
            pk  = np.roll(k, 1)
            ml, sl_, _ = macd(c, mf, ms)
            pml = np.roll(ml, 1); psl = np.roll(sl_, 1)
            lo  = (c > k) & (k > pk) & (ml > sl_) & (pml <= psl)
            sh  = (c < k) & (k < pk) & (ml < sl_) & (pml >= psl)
            return lo, sh
        systems.append(SignalDef(
            f"kama{klen}_macd{macd_fast}_{macd_slow}", {}, _kama_macd))

    # ── 20. SuperKAMA + ADX (Rust bota en yakın) ──
    for klen, adx_min in [(200, 20),(200,25),(500,20)]:
        def _sk_adx(h=h, l=l, c=c, kl=klen, am=adx_min):
            k   = kama(c, kl, 34, 55)
            ka  = atr(h, l, c, 14)
            pk  = np.roll(k,1); pc_ = np.roll(c,1)
            kr  = k > pk; kf = k < pk
            blo = k - ka; bhi = k + ka
            pbhi = np.roll(bhi,1); pblo = np.roll(blo,1)
            lo  = ((c > k) & (pc_ <= pk) & kr) | ((c > blo) & (pc_ <= pblo) & kr)
            sh  = ((c < k) & (pc_ >= pk) & kf) | ((c < bhi) & (pc_ >= pbhi) & kf)
            adx_, pdi, ndi = adx(h, l, c, 14)
            lo  = lo & (adx_ > am) & (pdi > ndi)
            sh  = sh & (adx_ > am) & (ndi > pdi)
            return lo, sh
        systems.append(SignalDef(
            f"sk{klen}_adx{adx_min}", {"kl":klen,"am":adx_min}, _sk_adx))

    return systems


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 4 — MATEMATİKSEL SL/TP OPTİMİZASYONU
# ══════════════════════════════════════════════════════════════════════════════

# ── Yardımcı: tek geçişte rolling quantile ──────────────────────────────────
def _rolling_q(arr: np.ndarray, win: int, q: float) -> np.ndarray:
    return (pd.Series(arr)
            .rolling(win, min_periods=max(1, win // 4))
            .quantile(q)
            .values)


def build_all_sltp(
    h: np.ndarray, l: np.ndarray, c: np.ndarray,
) -> list[tuple[str, np.ndarray, float]]:
    """
    Tüm (method, rr) kombinasyonları için sl_dist[i] dizisini ön-hesaplar.
    Returns: list of (name, sl_dist_array, rr)
      sl_dist[i] = i. mumda kullanılacak SL mesafesi (pozitif, fiyat birimi)
    """
    out: list[tuple[str, np.ndarray, float]] = []

    # ── A. ATR Yüzdelik ──────────────────────────────────────────────────────
    # sl_dist = rolling_quantile( ATR_full, win, q )
    for atr_p in [7, 14, 21]:
        a = atr(h, l, c, atr_p)          # tam dizi — tek hesap
        for (q_pct, win) in [(50, 50), (75, 50), (90, 50),
                             (75, 100), (90, 100)]:
            sl_arr = np.clip(_rolling_q(a, win, q_pct / 100.0), 1e-6, None)
            for rr in [1.5, 2.0, 2.5, 3.0]:
                out.append((f"atr{atr_p}_q{q_pct}w{win}_rr{rr}",
                             sl_arr.copy(), rr))

    # ── B. Fiyat Aralığı Yüzdelik ────────────────────────────────────────────
    # sl_dist = rolling_quantile( |delta_close|, win, q )
    delta = np.abs(np.diff(c, prepend=c[0]))          # len == n
    for win in [30, 60, 100]:
        for q_pct in [75, 90]:
            sl_arr = np.clip(_rolling_q(delta, win, q_pct / 100.0), 1e-6, None)
            for rr in [1.5, 2.0, 3.0]:
                out.append((f"range_w{win}_q{q_pct}_rr{rr}",
                             sl_arr.copy(), rr))

    # ── C. Swing High/Low ────────────────────────────────────────────────────
    # sl_dist ≈ yarı-swing aralığı * buffer
    for sw_lb in [5, 10, 20]:
        win = sw_lb * 2
        swh = pd.Series(h).rolling(win, min_periods=1).max().values
        swl = pd.Series(l).rolling(win, min_periods=1).min().values
        sl_arr = np.clip((swh - swl) * 0.51, 1e-6, None)
        for rr in [1.5, 2.0, 3.0]:
            out.append((f"swing_lb{sw_lb}_rr{rr}", sl_arr.copy(), rr))

    # ── D. Volatility Ratio Adaptif ──────────────────────────────────────────
    # sl_dist = fast_atr * (1 + max(0, 1 - fast/slow))
    for (fp, sp, rr) in [(5, 20, 2.0), (10, 50, 2.0),
                         (14, 50, 1.5), (7, 30, 2.5)]:
        fa   = atr(h, l, c, fp)
        sa   = atr(h, l, c, sp)
        ratio = fa / np.clip(sa, 1e-8, None)
        sl_arr = np.clip(fa * (1.0 + np.clip(1.0 - ratio, 0, None)), 1e-6, None)
        out.append((f"volratio_{fp}_{sp}_rr{rr}", sl_arr.copy(), float(rr)))

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 5 — BACKTEST MOTORU
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LabTrade:
    direction:   int
    entry_price: float
    entry_idx:   int
    sl:          float
    tp:          float
    orig_risk:   float = 0.0   # orijinal SL mesafesi (BE sonrası değişmez)
    result:      str   = ""
    exit_price:  float = 0.0
    exit_idx:    int   = 0
    pnl_r:       float = 0.0
    be_applied:  bool  = False

RISK_R = 1.0   # Sabit risk birimi


def run_backtest(
    long_sig:  np.ndarray,
    short_sig: np.ndarray,
    sl_dist:   np.ndarray,   # sl_dist[i] = i'de kullanılacak SL mesafesi
    rr:        float,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    be_candles: int = 8,
    max_hold:   int = 240,
    confirm_candle: bool = True,
) -> list[LabTrade]:

    trades: list[LabTrade] = []
    in_trade: Optional[LabTrade] = None
    pending  = None

    for i in range(len(c)):
        # ── Açık işlem yönetimi ──────────────────────────────────────────────
        if in_trade is not None:
            t = in_trade
            hi_, lo_, cl_ = h[i], l[i], c[i]
            held = i - t.entry_idx

            # BE
            if not t.be_applied and held >= be_candles:
                if t.direction == 1 and cl_ > t.entry_price:
                    t.sl = t.entry_price; t.be_applied = True
                elif t.direction == -1 and cl_ < t.entry_price:
                    t.sl = t.entry_price; t.be_applied = True

            closed = False
            risk_dist = max(t.orig_risk, 1e-10)  # orijinal SL mesafesi — BE'den etkilenmez

            # TP
            if t.direction == 1 and hi_ >= t.tp:
                t.exit_price = t.tp; t.exit_idx = i
                t.pnl_r = (t.tp - t.entry_price) / risk_dist * RISK_R
                t.result = "WIN"; closed = True
            elif t.direction == -1 and lo_ <= t.tp:
                t.exit_price = t.tp; t.exit_idx = i
                t.pnl_r = (t.entry_price - t.tp) / risk_dist * RISK_R
                t.result = "WIN"; closed = True

            # SL
            if not closed:
                if t.direction == 1 and lo_ <= t.sl:
                    t.exit_price = t.sl; t.exit_idx = i
                    t.result = "BE" if t.be_applied else "LOSS"
                    t.pnl_r  = 0.0 if t.be_applied else -RISK_R
                    closed = True
                elif t.direction == -1 and hi_ >= t.sl:
                    t.exit_price = t.sl; t.exit_idx = i
                    t.result = "BE" if t.be_applied else "LOSS"
                    t.pnl_r  = 0.0 if t.be_applied else -RISK_R
                    closed = True

            # Flip
            if not closed:
                if t.direction == 1 and short_sig[i]:
                    raw = (cl_ - t.entry_price) / risk_dist * RISK_R
                    t.exit_price = cl_; t.exit_idx = i; t.pnl_r = raw
                    t.result = "WIN" if raw > 0.05 else ("BE" if raw >= -0.05 else "LOSS")
                    closed = True
                elif t.direction == -1 and long_sig[i]:
                    raw = (t.entry_price - cl_) / risk_dist * RISK_R
                    t.exit_price = cl_; t.exit_idx = i; t.pnl_r = raw
                    t.result = "WIN" if raw > 0.05 else ("BE" if raw >= -0.05 else "LOSS")
                    closed = True

            # Max hold
            if not closed and held >= max_hold:
                raw = (cl_ - t.entry_price) * t.direction / risk_dist * RISK_R
                t.exit_price = cl_; t.exit_idx = i; t.pnl_r = raw
                t.result = "MAX"; closed = True

            if closed:
                trades.append(in_trade)
                in_trade = None

        # ── Pending onay ───────────────────────────────────────────────────
        if in_trade is None and pending is not None:
            prev_i  = pending["i"]
            direc   = pending["d"]
            entry   = c[i]   # sonraki bar açılışını temsilen kapanış kullanıyoruz
            _sd = float(sl_dist[min(prev_i, len(sl_dist) - 1)])
            if not (_sd > 0):   # handles nan, inf, zero
                _sd = entry * 0.0005
            d_val = max(_sd, entry * 0.0002, 1e-6)   # min 2 bps
            sl_   = entry - direc * d_val
            tp_   = entry + direc * d_val * rr
            if abs(entry - sl_) < 1e-8:
                pending = None
                continue
            in_trade = LabTrade(direc, entry, i, sl_, tp_, orig_risk=d_val)
            pending  = None

        # ── Yeni sinyal ────────────────────────────────────────────────────
        if in_trade is None and pending is None:
            d = 0
            if long_sig[i]:  d = 1
            elif short_sig[i]: d = -1
            if d != 0:
                if confirm_candle:
                    pending = {"i": i, "d": d}
                else:
                    _sd = float(sl_dist[min(i, len(sl_dist) - 1)])
                    if not (_sd > 0):   # handles nan, inf, zero
                        _sd = c[i] * 0.0005
                    d_val = max(_sd, c[i] * 0.0002, 1e-6)   # min 2 bps
                    sl_   = c[i] - d * d_val
                    tp_   = c[i] + d * d_val * rr
                    if abs(c[i] - sl_) < 1e-8:
                        continue
                    in_trade = LabTrade(d, c[i], i, sl_, tp_, orig_risk=d_val)

    return trades


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 6 — METRİKLER + WALK-FORWARD + MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LabResult:
    system:       str
    sltp_method:  str
    total_opened: int   = 0
    decisive:     int   = 0
    wins:         int   = 0
    losses:       int   = 0
    be_count:     int   = 0
    win_rate:     float = 0.0
    pnl_r:        float = 0.0
    avg_win:      float = 0.0
    avg_loss:     float = 0.0
    expectancy:   float = 0.0
    profit_factor:float = 0.0
    sharpe:       float = 0.0
    max_dd:       float = 0.0
    max_consec_l: int   = 0
    trades_per_day:float= 0.0
    # Walk-Forward
    wf_mean_wr:   float = 0.0
    wf_std_wr:    float = 0.0
    wf_positive_pct:float= 0.0
    # Monte Carlo
    mc_pnl_p10:   float = 0.0
    mc_pnl_p50:   float = 0.0
    mc_pnl_p90:   float = 0.0
    mc_ruin_pct:  float = 0.0   # Equity 0'ın altına düşme oranı


def calc_metrics(trades: list[LabTrade], days: int,
                 system: str, sltp: str) -> LabResult:
    r = LabResult(system=system, sltp_method=sltp)
    r.total_opened = len(trades)
    if not trades:
        return r

    decisive  = [t for t in trades if t.result in ("WIN","LOSS")]
    be_trades = [t for t in trades if t.result == "BE"]
    r.be_count = len(be_trades)

    if not decisive:
        return r

    wins   = [t for t in decisive if t.result == "WIN"]
    losses = [t for t in decisive if t.result == "LOSS"]
    r.decisive  = len(decisive)
    r.wins      = len(wins)
    r.losses    = len(losses)
    r.win_rate  = r.wins / r.decisive * 100

    r.avg_win   = np.mean([t.pnl_r for t in wins])   if wins   else 0.0
    r.avg_loss  = np.mean([t.pnl_r for t in losses]) if losses else 0.0
    gw = sum(t.pnl_r for t in wins)
    gl = abs(sum(t.pnl_r for t in losses))
    r.pnl_r          = sum(t.pnl_r for t in trades)
    r.expectancy      = r.win_rate/100 * r.avg_win + (1-r.win_rate/100) * r.avg_loss
    r.profit_factor   = gw / gl if gl > 0 else float("inf")

    pnls = np.array([t.pnl_r for t in decisive])
    yearly_trades = len(decisive) / max(days, 1) * 252
    ann_f = math.sqrt(max(yearly_trades, 1))
    r.sharpe = (np.mean(pnls) / np.std(pnls) * ann_f) if np.std(pnls) > 0 else 0

    equity = np.cumsum([t.pnl_r for t in trades])
    peak   = np.maximum.accumulate(equity)
    r.max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

    best = cur = 0
    for t in decisive:
        if t.result == "LOSS": cur += 1; best = max(best, cur)
        else: cur = 0
    r.max_consec_l  = best
    r.trades_per_day = r.decisive / max(days, 1)
    return r


def walk_forward(
    long_sig, short_sig, sl_dist: np.ndarray, rr: float, h, l, c,
    n_splits: int = 5, be_candles: int = 8,
) -> tuple[float, float, float]:
    """Slice → her dilimde WR hesapla → mean/std/positive_pct döndür."""
    chunk = len(c) // n_splits
    wrs   = []
    for s in range(n_splits):
        i0, i1 = s*chunk, (s+1)*chunk
        trades = run_backtest(
            long_sig[i0:i1], short_sig[i0:i1],
            sl_dist[i0:i1], rr,
            h[i0:i1], l[i0:i1], c[i0:i1],
            be_candles=be_candles, confirm_candle=True)
        dec = [t for t in trades if t.result in ("WIN","LOSS")]
        if dec:
            wrs.append(sum(1 for t in dec if t.result=="WIN")/len(dec)*100)
    if not wrs:
        return 0.0, 0.0, 0.0
    return float(np.mean(wrs)), float(np.std(wrs)), float(np.mean([w>50 for w in wrs]))


def monte_carlo(trades: list[LabTrade], n_sim: int = 300,
                ruin_threshold: float = -20.0) -> tuple[float,float,float,float]:
    """Shuffle trades N kez → PNL dağılımı + ruin probability."""
    if not trades:
        return 0.0, 0.0, 0.0, 1.0
    pnls = np.array([t.pnl_r for t in trades])
    final_pnls = []
    ruin_count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_sim):
        shuffled = rng.choice(pnls, size=len(pnls), replace=True)
        equity   = np.cumsum(shuffled)
        final_pnls.append(equity[-1])
        if equity.min() < ruin_threshold:
            ruin_count += 1
    arr = np.array(final_pnls)
    return (float(np.percentile(arr, 10)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
            ruin_count / n_sim)


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 7 — ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    symbol:     str,
    days:       int,
    force_dl:   bool,
    system_filter: list[str],
    top_n:      int = 30,
    be_candles: int = 8,
    mc_sims:    int = 200,
    wf_splits:  int = 4,
) -> list[LabResult]:

    # Veri
    print(f"\n{'='*64}")
    print(f"  EXPERIMENT LAB  --  {symbol} 1m  {days}g")
    print(f"{'='*64}")
    print("[1] Veri yukleniyor...")
    df = load_or_download(symbol, days, force_dl)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    print(f"  Toplam mum: {len(df):,}  ({df['open_time'].iloc[0].date()} -> {df['open_time'].iloc[-1].date()})")

    # Sinyal sistemleri
    print("[2] Sinyal sistemleri olusturuluyor...")
    all_systems = _make_systems(df)
    if system_filter:
        all_systems = [s for s in all_systems
                       if any(f in s.name for f in system_filter)]
    print(f"  {len(all_systems)} sistem x SL/TP yontemleri")

    # SL/TP yöntemleri
    print("[3] SL/TP yontemleri hazirlaniyor...")
    all_sltp = build_all_sltp(h, l, c)
    print(f"  {len(all_sltp)} SL/TP yontemi")

    total_combos = len(all_systems) * len(all_sltp)
    print(f"\n[4] Toplam kombinasyon: {total_combos:,}")
    print(f"  (BE={be_candles} mum, WF={wf_splits} dilim, MC={mc_sims} sim)\n")

    all_results: list[LabResult] = []
    combo_idx   = 0
    t0          = time.time()

    for sys_def in all_systems:
        # Sistemi bir kez hesapla
        try:
            long_s, short_s = sys_def.generate()
        except Exception as e:
            continue

        n_sigs = long_s.sum() + short_s.sum()
        if n_sigs < 10:
            continue   # Çok az sinyal — atla

        for sltp_name, sl_dist, rr in all_sltp:
            combo_idx += 1

            # Hız takibi
            if combo_idx % 200 == 0:
                elapsed = time.time() - t0
                rate    = combo_idx / elapsed
                eta     = (total_combos - combo_idx) / max(rate, 1)
                print(f"  [{combo_idx:>5}/{total_combos}]  "
                      f"{elapsed:.0f}s gecti  ETA ~{eta:.0f}s"
                      f"  top WR: "
                      f"{max((r.win_rate for r in all_results), default=0):.1f}%  "
                      f"top EV: {max((r.expectancy for r in all_results), default=0):+.3f}R")

            trades = run_backtest(
                long_s, short_s, sl_dist, rr, h, l, c,
                be_candles=be_candles, confirm_candle=True)

            res = calc_metrics(trades, days, sys_def.name, sltp_name)

            # Anlamsız sonuçları eliyle
            if res.decisive < 20:
                continue

            # Walk-Forward
            wf_wr, wf_std, wf_pos = walk_forward(
                long_s, short_s, sl_dist, rr, h, l, c,
                n_splits=wf_splits, be_candles=be_candles)
            res.wf_mean_wr       = wf_wr
            res.wf_std_wr        = wf_std
            res.wf_positive_pct  = wf_pos

            # Monte Carlo (sadece umut verici kombinasyonlar için)
            if res.expectancy > 0 and res.profit_factor > 1.05:
                p10, p50, p90, ruin = monte_carlo(trades, mc_sims)
                res.mc_pnl_p10  = p10
                res.mc_pnl_p50  = p50
                res.mc_pnl_p90  = p90
                res.mc_ruin_pct = ruin

            all_results.append(res)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 8 — RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def _rank_score(r: LabResult) -> float:
    """Çok boyutlu sıralama skoru."""
    if r.decisive < 20 or r.profit_factor <= 0:
        return -999
    # Bileşik skor: EV × Sharpe × WF_positive × (1-ruin)
    score = (r.expectancy * max(r.sharpe, 0) *
             max(r.wf_positive_pct, 0.01) *
             max(1 - r.mc_ruin_pct, 0.01))
    return score


def print_top(results: list[LabResult], n: int = 20,
              label: str = "GENEL SKOR"):
    ranked = sorted(results, key=_rank_score, reverse=True)[:n]
    print(f"\n{'='*72}")
    print(f"  TOP {n}  --  {label}")
    print(f"{'='*72}")
    header = (f"{'#':>3}  {'Sistem':<28}  {'SL/TP':<30}  "
              f"{'D':>4}  {'WR%':>5}  {'EV':>6}  "
              f"{'PF':>5}  {'Sharpe':>6}  {'DD':>5}  "
              f"{'WF%':>5}  {'MC50':>6}  {'Ruin%':>5}")
    print(header)
    print("-"*len(header))
    for idx, r in enumerate(ranked, 1):
        print(f"{idx:>3}  {r.system:<28}  {r.sltp_method:<30}  "
              f"{r.decisive:>4}  {r.win_rate:>5.1f}  {r.expectancy:>+6.3f}  "
              f"{r.profit_factor:>5.2f}  {r.sharpe:>6.2f}  {r.max_dd:>5.2f}  "
              f"{r.wf_positive_pct*100:>5.0f}  {r.mc_pnl_p50:>+6.1f}  {r.mc_ruin_pct*100:>5.1f}")


def save_all(results: list[LabResult], symbol: str):
    out = [asdict(r) for r in results]
    # Skor ekle
    for o, r in zip(out, results):
        o["rank_score"] = _rank_score(r)
    out.sort(key=lambda x: x["rank_score"], reverse=True)
    path = LAB_DIR / f"{symbol}_1m_lab.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  OK Tum sonuclar -> {path}")

    # CSV de kaydet
    df = pd.DataFrame(out)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"  OK CSV -> {path.with_suffix('.csv').name}")
    return path


def show_summary(symbol: str):
    """Daha önce kaydedilmiş sonuçları listele."""
    path = LAB_DIR / f"{symbol}_1m_lab.json"
    if not path.exists():
        print(f"  Sonuc bulunamadi: {path}  (once --symbol {symbol} ile calistir)")
        return
    with open(path) as f:
        data = json.load(f)
    results = [LabResult(**{k:v for k,v in d.items() if k != "rank_score"})
               for d in data]
    for r, d in zip(results, data):
        r.__dict__["_score"] = d.get("rank_score", 0)
    print_top(results, n=30, label=f"{symbol} 1m -- Kayitli Sonuclar")


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",   default="ETHUSDT")
    p.add_argument("--days",     type=int, default=365)
    p.add_argument("--download", action="store_true")
    p.add_argument("--systems",  default="",
                   help="Virgülle: superkama,ema,macd,rsi,bb,stochrsi,adx,supertrend,vwap,obv,pivot")
    p.add_argument("--summary",  action="store_true", help="Kayıtlı sonuçları göster")
    p.add_argument("--top",      type=int, default=20)
    p.add_argument("--be",       type=int, default=8,   help="BE candle sayısı")
    p.add_argument("--mc",       type=int, default=200, help="Monte Carlo simülasyon sayısı")
    p.add_argument("--wf",       type=int, default=4,   help="Walk-Forward dilim sayısı")
    args = p.parse_args()

    symbol = args.symbol.upper()

    if args.summary:
        show_summary(symbol)
        return

    sys_filter = [s.strip() for s in args.systems.split(",") if s.strip()]

    results = run_experiment(
        symbol      = symbol,
        days        = args.days,
        force_dl    = args.download,
        system_filter = sys_filter,
        top_n       = args.top,
        be_candles  = args.be,
        mc_sims     = args.mc,
        wf_splits   = args.wf,
    )

    if not results:
        print("\n  !!  Yeterli sonuc uretilemedi. Daha uzun veri veya farkli sistem dene.")
        return

    print_top(results, n=args.top, label="GENEL SKOR")

    # Kategorik en iyiler
    def _top1(key, reverse=True):
        f = sorted([r for r in results if r.decisive >= 20],
                   key=lambda r: getattr(r, key), reverse=reverse)
        return f[0] if f else None

    cats = [
        ("win_rate",      "En Yüksek Win Rate"),
        ("sharpe",        "En Yüksek Sharpe"),
        ("expectancy",    "En Yüksek EV/işlem"),
        ("pnl_r",         "En Yüksek Toplam PNL (R)"),
        ("wf_positive_pct","En Tutarlı (Walk-Forward)"),
    ]
    print(f"\n{'='*72}")
    print("  KATEGORIK KAZANANLAR")
    print(f"{'='*72}")
    for key, label in cats:
        best = _top1(key)
        if best:
            print(f"\n  >> {label}")
            print(f"    Sistem   : {best.system}")
            print(f"    SL/TP    : {best.sltp_method}")
            print(f"    WR       : {best.win_rate:.1f}%  |  EV: {best.expectancy:+.4f}R  "
                  f"|  PF: {best.profit_factor:.3f}  |  Sharpe: {best.sharpe:.2f}")
            print(f"    MC P50   : {best.mc_pnl_p50:+.1f}R  "
                  f"|  Ruin: {best.mc_ruin_pct*100:.1f}%  "
                  f"|  WF_pos: {best.wf_positive_pct*100:.0f}%")

    save_all(results, symbol)
    print("\n  OK Tamamlandi.\n")


if __name__ == "__main__":
    main()
