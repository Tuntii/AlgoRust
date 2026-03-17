import json
from datetime import datetime, timedelta

data = json.load(open('backtest_results/ETHUSDT_1m_backtest.json'))
signals = data['signals']

# Total candles in 7 days of 1m: ~10080
# End time is approximately 2026-03-17 14:15 UTC (when backtest ran)
# Build candle time mapper: candle_idx -> approximate datetime
total_candles = data.get('total_trades', 0)  # fallback
# The backtest fetched ~10080 candles (7 days * 1440 min/day)
# Ending around 2026-03-17T14:14 UTC
end_time = datetime(2026, 3, 17, 14, 14)
# Find max candle_idx to estimate start
max_idx = max(t.get('entry_candle_idx', 0) for t in signals)
# Candles start from index 0 at roughly end_time - total_candlesmin
# Total candles fetched: let's estimate from the log (10081 evaluations, ~3000 bootstrap)
total_fetched = 10081  # from BACKTEST_SUMMARY block_stats.total_evaluations
start_time = end_time - timedelta(minutes=total_fetched)

def idx_to_time(idx):
    return start_time + timedelta(minutes=idx)

# ======= LIVE TRADES from PDF ========
live_trades = [
    ("2026-03-16 17:43", "Short", 2259.34, 2276.00, -0.84966),
    ("2026-03-16 11:53", "Short", 2243.28, 2247.89, -0.23511),
    ("2026-03-16 11:46", "Short", 2239.55, 2243.19, -0.18928),
    ("2026-03-16 05:23", "Short", 2175.74, 2186.19, -0.5643),
    ("2026-03-16 05:21", "Short", 2177.52, 2175.75, 0.09558),
    ("2026-03-16 05:20", "Short", 2175.66, 2177.38, -0.09288),
    ("2026-03-16 05:18", "Short", 2178.45, 2176.24, 0.11934),
    ("2026-03-16 05:17", "Short", 2176.84, 2178.46, -0.08748),
    ("2026-03-16 04:24", "Long", 2177.90, 2180.80, 0.1566),
    ("2026-03-16 03:07", "Long", 2181.40, 2171.16, -0.55296),
    ("2026-03-16 03:06", "Long", 2183.00, 2181.45, -0.08525),
    ("2026-03-12 21:06", "Short", 2053.30, 2045.75, 0.55115),
    ("2026-03-12 18:25", "Long", 2053.51, 2067.03, 0.97344),
    ("2026-03-12 18:19", "Long", 2049.27, 2053.85, 0.32976),
    ("2026-03-12 17:33", "Short", 2050.89, 2048.99, 0.1368),
    ("2026-03-12 04:03", "Short", 2048.13, 2029.91, 1.38472),
    ("2026-03-12 03:38", "Short", 2052.14, 2056.65, -0.34727),
    ("2026-03-12 03:36", "Short", 2052.98, 2051.86, 0.08624),
    ("2026-03-12 03:03", "Short", 2049.86, 2052.99, -0.24101),
    ("2026-03-12 02:49", "Short", 2048.12, 2049.87, -0.1365),
    ("2026-03-11 18:32", "Short", 2029.97, 2061.17, -2.5272),
    ("2026-03-11 15:38", "Long", 2022.41, 2030.14, 0.62613),
    ("2026-03-11 14:29", "Short", 2022.28, 2013.34, 0.7152),
    ("2026-03-11 11:44", "Long", 2022.65, 2022.17, -0.03888),
    ("2026-03-11 08:30", "Short", 2031.30, 2020.81, 0.8392),
    ("2026-03-11 04:55", "Long", 2035.40, 2022.85, -1.01623),
    ("2026-03-11 04:23", "Short", 2033.29, 2035.16, -0.15147),
    ("2026-03-11 03:10", "Short", 2035.21, 2038.18, -0.24057),
    ("2026-03-10 21:16", "Short", 2051.46, 2020.94, 2.41108),
    ("2026-03-10 04:39", "Short", 2007.07, 2021.96, -1.22098),
    ("2026-03-10 01:53", "Short", 2001.33, 2013.12, -0.97857),
]

print("=" * 120)
print("LIVE TRADES (Binance Futures PDF) — 31 trades")
print("=" * 120)
live_wins = sum(1 for t in live_trades if t[4] > 0)
live_losses = sum(1 for t in live_trades if t[4] <= 0)
live_pnl = sum(t[4] for t in live_trades)
print(f"Wins: {live_wins}, Losses: {live_losses}, Win Rate: {live_wins/len(live_trades)*100:.1f}%, Net PnL: {live_pnl:+.5f} USDT")
print()
for t in live_trades:
    pnl_str = f"{t[4]:+.5f}"
    result = "WIN" if t[4] > 0 else "LOSS"
    print(f"  {t[0]} | {t[1]:5s} | entry={t[2]:.2f} exit={t[3]:.2f} | pnl={pnl_str} | {result}")

print()
print("=" * 120)
print("BACKTEST TRADES (indicator_flip mode, 7 days) — decisive only")
print("=" * 120)

bt_decisive = []
bt_be = []
for t in signals:
    if not t.get('outcome'):
        continue
    outcome = t['outcome']
    sig = t['signal']
    entry = float(t['entry_price'])
    sl = float(t['sl_price'])
    tp = float(t['tp_price'])
    exit_p = float(t['exit_price']) if t.get('exit_price') else 0
    pnl_r = float(t['pnl_r']) if t.get('pnl_r') else 0
    idx = t.get('entry_candle_idx', 0)
    est_time = idx_to_time(idx)
    direction = sig['direction']
    dur = t.get('duration_candles', 0)
    
    row = (est_time, direction, entry, sl, tp, exit_p, pnl_r, outcome, dur, 
           t.get('partial_tp_taken', False), t.get('is_be_applied', False))
    
    if outcome in ('WIN', 'LOSS', 'TRAIL_TP'):
        bt_decisive.append(row)
    else:
        bt_be.append(row)

bt_wins = sum(1 for r in bt_decisive if r[7] in ('WIN', 'TRAIL_TP') and r[6] > 0)
bt_losses = sum(1 for r in bt_decisive if r[7] == 'LOSS' or r[6] < 0)
bt_pnl = sum(r[6] for r in bt_decisive)
print(f"Decisive: {len(bt_decisive)} (Wins: {bt_wins}, Losses: {bt_losses}), BE exits: {len(bt_be)}, Total opened: {len(signals)}")
print(f"Win Rate: {bt_wins/max(len(bt_decisive),1)*100:.1f}%, Net PnL (R): {bt_pnl:+.3f}")
print()
for r in bt_decisive:
    flags = ""
    if r[9]: flags += " PT"
    if r[10]: flags += " BE"
    print(f"  ~{r[0].strftime('%m-%d %H:%M')} | {r[1]:5s} | entry={r[2]:.2f} sl={r[3]:.2f} tp={r[4]:.2f} | exit={r[5]:.2f} | pnl_r={r[6]:+.3f} | {r[7]:10s} | dur={r[8]:>3}{flags}")

print()
print("=" * 120)
print("CRITICAL DIFFERENCES: BACKTEST vs LIVE")
print("=" * 120)
print("""
1. EXIT MODE MISMATCH (EN KRİTİK FARK):
   - Backtest: exit_mode = "indicator_flip" (karşı sinyal gelince kapanır, TP yok)
   - Live:     live_exit_mode = "sl_tp"     (SL + TP emri koyar, sinyal çıkışı yok)
   → Backtest'te trade'ler karşı sinyal gelene kadar açık kalıyor (trailing stop aktif)
   → Live'da TP order'ı hemen tetiklenebiliyor veya SL'ye takılıyor

2. ENTRY TIMING:
   - Backtest: entry = candle.close (aynı bar — lookahead bias riski)
   - Live:     entry = next bar open (pending_signal mekanizması ile)
   → Backtest entry fiyatı, live'dan 1 mum erken

3. SL/TP HESAPLAMA:
   - Backtest: calculate_sl_tp() fonksiyonu (runner.rs) — OB SL/TP
   - Live:     ctx.ob_tracker.calculate_ob_sl_tp() + atr_scale (binance_trader.rs)
   → Her ikisi de OB tabanlı ama backtest'te farklı SL seviyesine BE yapıyor

4. BE MEKANİZMASI:
   - Backtest: be_threshold=7 candles, be_min_profit_r=0.1 → SL'yi entry'ye çeker
   - Live:     BE mekanizması YOK (sadece SL/TP order'ları var)
   → Backtest'te 44 trade BE ile kapandı (gerçekte bunlar SL veya TP olurdu)

5. PARTIAL TP:
   - Backtest: 1R'da %50 partial TP alıyor → 0.5R kilitliyor  
   - Live:     Partial TP YOK — ya full SL ya full TP
   → Backtest kârları şişiriyor

6. TRAILING STOP:
   - Backtest: 0.4% callback, 35% activation — 5 trade TRAIL_TP ile kapandı
   - Live:     trailing_stop_enabled = false (config'de kapalı)
   → Backtest'in trailing TP kârları live'da gerçekleşmiyor

7. MULTI-POSITION:
   - Backtest: max_active_trades = 3 (aynı anda 3 pozisyon)
   - Live:     max_positions = 1 (tek pozisyon)
   → Backtest daha fazla sinyal alıp diversifiye ediyor

8. COOLDOWN:  
   - Backtest: new_backtest_mode() — farklı cooldown'lar
   - Live:     PolicyEngine::new() — standart cooldown'lar
""")

# Overlapping period analysis
print("=" * 120)
print("PRICE LEVEL MATCHING (Live entry fiyatları backtest'te var mı?)")
print("=" * 120)
for lt in live_trades:
    live_entry = lt[2]
    matches = []
    for bt in bt_decisive + bt_be:
        if abs(bt[2] - live_entry) < 3.0:  # Within $3
            matches.append(bt)
    
    match_str = ""
    if matches:
        for m in matches[:2]:
            match_str += f" → BT ~{m[0].strftime('%m-%d %H:%M')} {m[1]} entry={m[2]:.2f} {m[7]}"
    else:
        match_str = " → NO MATCH in backtest"
    
    print(f"  Live {lt[0]} {lt[1]:5s} entry={live_entry:.2f} pnl={lt[4]:+.5f}{match_str}")
