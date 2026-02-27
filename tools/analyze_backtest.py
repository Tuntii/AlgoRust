#!/usr/bin/env python3
"""Backtest JSON analiz scripti - outcome, yön, süre, pnl dağılımı."""

import json, os, statistics
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest_results")

files = {
    "ETHUSDT": "ETHUSDT_1m_backtest.json",
    "SOLUSDT": "SOLUSDT_1m_backtest.json",
}

for sym, fname in files.items():
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
        data = json.load(f)

    signals = data.get("signals", [])
    print(f"\n{'='*60}")
    print(f"  {sym}  — {len(signals)} işlem")
    print(f"{'='*60}")

    # --- Outcome dağılımı ---
    outcomes = Counter(s.get("outcome", "?") for s in signals)
    print("\n[Outcome Dağılımı]")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        pct = v / len(signals) * 100
        print(f"  {k:<15} {v:>4}  ({pct:.1f}%)")

    # --- Yön dağılımı ---
    dirs = Counter(s.get("direction", "?") for s in signals)
    print("\n[Yön Dağılımı]")
    for k, v in dirs.items():
        pct = v / len(signals) * 100
        print(f"  {k:<10} {v:>4}  ({pct:.1f}%)")

    # --- Kazanç / Kayıp ortalama pnl_r ---
    wins   = [float(s["pnl_r"]) for s in signals if float(s.get("pnl_r", 0)) > 0]
    losses = [float(s["pnl_r"]) for s in signals if float(s.get("pnl_r", 0)) < 0]
    bes    = [s for s in signals if s.get("outcome") == "BE"]
    maxdur = [s for s in signals if s.get("outcome") == "MaxDuration"]
    print(f"\n[PnL_R]")
    print(f"  Kazanç sayısı : {len(wins):<5}  ort={statistics.mean(wins):.3f}R  "
          f"medyan={statistics.median(wins):.3f}R" if wins else "  Kazanç yok")
    print(f"  Kayıp sayısı  : {len(losses):<5}  ort={statistics.mean(losses):.3f}R" if losses else "  Kayıp yok")
    print(f"  BE sayısı     : {len(bes):<5}")
    print(f"  MaxDuration   : {len(maxdur):<5}")

    # --- Süre analizi ---
    dur_w  = [s["duration_candles"] for s in signals if float(s.get("pnl_r", 0)) > 0]
    dur_l  = [s["duration_candles"] for s in signals if float(s.get("pnl_r", 0)) < 0]
    dur_be = [s["duration_candles"] for s in signals if s.get("outcome") == "BE"]
    print(f"\n[Ortalama Süre (mum)]")
    if dur_w:  print(f"  Kazanç  : {statistics.mean(dur_w):.1f} bars (medyan {statistics.median(dur_w):.0f})")
    if dur_l:  print(f"  Kayıp   : {statistics.mean(dur_l):.1f} bars (medyan {statistics.median(dur_l):.0f})")
    if dur_be: print(f"  BE      : {statistics.mean(dur_be):.1f} bars (medyan {statistics.median(dur_be):.0f})")

    # --- LONG vs SHORT performans ---
    print(f"\n[LONG vs SHORT Win Rate]")
    for direction in ["LONG", "SHORT"]:
        sub = [s for s in signals if s.get("direction") == direction]
        if not sub: continue
        w = sum(1 for s in sub if float(s.get("pnl_r", 0)) > 0)
        l = sum(1 for s in sub if float(s.get("pnl_r", 0)) < 0)
        pnl = sum(float(s.get("pnl_r", 0)) for s in sub)
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        print(f"  {direction:<6}: {len(sub):>3} işlem | {w}W/{l}L | WR={wr:.1f}% | PnL={pnl:.2f}R")

    # --- En kötü kayıplar ---
    sorted_by_loss = sorted(signals, key=lambda s: float(s.get("pnl_r", 0)))[:5]
    print(f"\n[En Büyük 5 Kayıp]")
    for s in sorted_by_loss:
        if float(s.get("pnl_r", 0)) >= 0:
            break
        print(f"  {s.get('direction','?'):<6} | entry={s.get('entry_price','?'):>10} | "
              f"pnl={float(s.get('pnl_r',0)):>7.3f}R | dur={s.get('duration_candles','?'):>3} bars | "
              f"EMA_slope={float(s.get('ema50_slope_at_entry',0)):.6f}")

    # --- MaxDuration işlemlere bak ---
    if maxdur:
        pnl_list = [float(s.get("pnl_r", 0)) for s in maxdur]
        neg = sum(1 for p in pnl_list if p < 0)
        zero = sum(1 for p in pnl_list if p == 0)
        pos = sum(1 for p in pnl_list if p > 0)
        print(f"\n[MaxDuration Çıkış Analizi]")
        print(f"  Toplam: {len(maxdur)} | Kazanç: {pos} | Kayıp: {neg} | BE: {zero}")
        print(f"  Ort PnL: {statistics.mean(pnl_list):.3f}R")

    # --- EMA50 slope histogram (kayıplarda) ---
    if losses:
        loss_sigs = [s for s in signals if float(s.get("pnl_r", 0)) < 0]
        slopes = [abs(float(s.get("ema50_slope_at_entry", 0))) for s in loss_sigs]
        small = sum(1 for x in slopes if x < 0.0002)
        med   = sum(1 for x in slopes if 0.0002 <= x < 0.0005)
        large = sum(1 for x in slopes if x >= 0.0005)
        print(f"\n[Kayıplardaki EMA50 Slope Dağılımı]")
        print(f"  <0.0002 (düşük eğim) : {small}")
        print(f"  0.0002-0.0005        : {med}")
        print(f"  >=0.0005 (yüksek)    : {large}")

print("\nAnaliz tamamlandı.\n")
