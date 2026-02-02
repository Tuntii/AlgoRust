#!/usr/bin/env python3
"""
Backtest Visualization Tool
===========================
Bu script backtest sonuçlarını görselleştirir.
Entry/Exit noktalarını, SL/TP seviyelerini candlestick chart üzerinde gösterir.

Kullanım:
    python visualize_backtest.py                           # Interaktif menü
    python visualize_backtest.py --file BTCUSDT_1h        # Belirli dosya
    python visualize_backtest.py --file BTCUSDT_1h --trade 0  # Belirli trade
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import requests
except ImportError as e:
    print(f"❌ Gerekli kütüphaneler eksik: {e}")
    print("\n📦 Yüklemek için:")
    print("   pip install pandas numpy mplfinance matplotlib requests")
    sys.exit(1)


# =============================================================================
# BINANCE DATA FETCHER
# =============================================================================

def fetch_binance_candles(symbol: str, interval: str, limit: int = 1000, end_time: int = None) -> pd.DataFrame:
    """Binance'den candlestick verisi çeker."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('open_time', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"❌ Binance API hatası: {e}")
        return pd.DataFrame()


# =============================================================================
# BACKTEST LOADER
# =============================================================================

def load_backtest_results(filepath: str) -> dict:
    """Backtest JSON dosyasını yükler."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_backtest_files(directory: str = "backtest_results") -> list:
    """Mevcut backtest dosyalarını listeler."""
    files = []
    for f in Path(directory).glob("*_backtest.json"):
        files.append(f.stem.replace("_backtest", ""))
    return sorted(files)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_trade_markers(trades: list, df: pd.DataFrame) -> tuple:
    """Trade'ler için marker pozisyonları oluşturur."""
    long_entries = []
    long_exits = []
    short_entries = []
    short_exits = []
    
    for trade in trades:
        if trade.get('outcome') is None:
            continue
            
        entry_idx = trade.get('entry_candle_idx', 0)
        exit_idx = trade.get('exit_candle_idx', entry_idx)
        
        if entry_idx >= len(df) or exit_idx >= len(df):
            continue
        
        entry_price = float(trade['entry_price'])
        exit_price = float(trade.get('exit_price', entry_price))
        direction = trade['direction']
        
        if direction == "LONG":
            long_entries.append((entry_idx, entry_price))
            long_exits.append((exit_idx, exit_price))
        else:
            short_entries.append((entry_idx, entry_price))
            short_exits.append((exit_idx, exit_price))
    
    return long_entries, long_exits, short_entries, short_exits


def visualize_single_trade(trade: dict, df: pd.DataFrame, symbol: str, timeframe: str, trade_num: int):
    """Tek bir trade'i detaylı görselleştirir."""
    
    entry_idx = trade.get('entry_candle_idx', 0)
    exit_idx = trade.get('exit_candle_idx', entry_idx + 10)
    
    # Trade etrafında context göster (önce 20, sonra 30 candle)
    start_idx = max(0, entry_idx - 20)
    end_idx = min(len(df), exit_idx + 30)
    
    trade_df = df.iloc[start_idx:end_idx].copy()
    
    if len(trade_df) == 0:
        print(f"⚠️ Trade #{trade_num} için veri bulunamadı")
        return
    
    # Trade bilgileri
    entry_price = float(trade['entry_price'])
    sl_price = float(trade.get('original_sl_price', trade['sl_price']))
    tp_price = float(trade['tp_price'])
    exit_price = float(trade.get('exit_price', entry_price))
    direction = trade['direction']
    outcome = trade.get('outcome', 'OPEN')
    pnl_r = trade.get('pnl_r', 0)
    confidence = trade.get('adjusted_confidence', trade.get('confidence', 0))
    duration = trade.get('duration_candles', 0)
    
    # Renk belirleme
    if outcome == "WIN":
        outcome_color = '#00C853'  # Yeşil
    elif outcome == "LOSS":
        outcome_color = '#FF1744'  # Kırmızı
    elif outcome == "BE":
        outcome_color = '#FFD600'  # Sarı
    else:
        outcome_color = '#2196F3'  # Mavi
    
    # Horizontal lines
    hlines = dict(
        hlines=[entry_price, sl_price, tp_price],
        colors=['#2196F3', '#FF1744', '#00C853'],
        linestyle=['--', '-.', '-.'],
        linewidths=[1.5, 1, 1]
    )
    
    # Entry/Exit markers - relative index hesapla
    entry_rel_idx = entry_idx - start_idx
    exit_rel_idx = exit_idx - start_idx if exit_idx else entry_rel_idx
    
    # Marker arrays
    entry_markers = [np.nan] * len(trade_df)
    exit_markers = [np.nan] * len(trade_df)
    
    if 0 <= entry_rel_idx < len(trade_df):
        entry_markers[entry_rel_idx] = entry_price
    if 0 <= exit_rel_idx < len(trade_df):
        exit_markers[exit_rel_idx] = exit_price
    
    # Add plot için marker series
    apds = []
    
    if direction == "LONG":
        entry_marker = mpf.make_addplot(
            entry_markers, type='scatter', markersize=200, 
            marker='^', color='#00C853'
        )
        exit_marker = mpf.make_addplot(
            exit_markers, type='scatter', markersize=200,
            marker='v', color=outcome_color
        )
    else:
        entry_marker = mpf.make_addplot(
            entry_markers, type='scatter', markersize=200,
            marker='v', color='#FF1744'
        )
        exit_marker = mpf.make_addplot(
            exit_markers, type='scatter', markersize=200,
            marker='^', color=outcome_color
        )
    
    apds.extend([entry_marker, exit_marker])
    
    # Reasons'ı al
    reasons = trade.get('reason', [])
    reasons_text = '\n'.join(reasons[:5]) if reasons else 'N/A'
    
    # Title
    title = f"{symbol} {timeframe} | Trade #{trade_num} | {direction} | {outcome}\n"
    title += f"Entry: ${entry_price:,.2f} | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}\n"
    title += f"PnL: {pnl_r}R | Confidence: {confidence} | Duration: {duration} candles"
    
    # Style
    mc = mpf.make_marketcolors(
        up='#00C853', down='#FF1744',
        edge={'up': '#00C853', 'down': '#FF1744'},
        wick={'up': '#00C853', 'down': '#FF1744'},
        volume='#64B5F6'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#333333',
        facecolor='#1a1a2e',
        figcolor='#1a1a2e',
        rc={
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white'
        }
    )
    
    # Plot
    fig, axes = mpf.plot(
        trade_df,
        type='candle',
        style=style,
        title=title,
        ylabel='Price ($)',
        volume=True,
        hlines=hlines,
        addplot=apds,
        figsize=(16, 10),
        returnfig=True,
        tight_layout=True
    )
    
    # Reasons annotation
    ax = axes[0]
    ax.annotate(
        reasons_text,
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        fontsize=8,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#0f3460', alpha=0.9),
        color='white'
    )
    
    # SL/TP Zone shading
    if direction == "LONG":
        # TP zone (yeşil)
        ax.axhspan(entry_price, tp_price, alpha=0.1, color='#00C853')
        # SL zone (kırmızı)
        ax.axhspan(sl_price, entry_price, alpha=0.1, color='#FF1744')
    else:
        # TP zone (yeşil)
        ax.axhspan(tp_price, entry_price, alpha=0.1, color='#00C853')
        # SL zone (kırmızı)
        ax.axhspan(entry_price, sl_price, alpha=0.1, color='#FF1744')
    
    plt.show()


def visualize_all_trades(trades: list, df: pd.DataFrame, symbol: str, timeframe: str, result: dict):
    """Tüm trade'leri özet olarak görselleştirir."""
    
    # Trade statistics
    wins = result.get('wins', 0)
    losses = result.get('losses', 0)
    total = result.get('total_trades', len(trades))
    win_rate = result.get('win_rate', 0)
    pnl = result.get('total_pnl_r', 0)
    
    # Marker arrays
    long_entry = [np.nan] * len(df)
    long_exit_win = [np.nan] * len(df)
    long_exit_loss = [np.nan] * len(df)
    short_entry = [np.nan] * len(df)
    short_exit_win = [np.nan] * len(df)
    short_exit_loss = [np.nan] * len(df)
    
    for trade in trades:
        if trade.get('outcome') is None:
            continue
            
        entry_idx = trade.get('entry_candle_idx', 0)
        exit_idx = trade.get('exit_candle_idx', entry_idx)
        
        if entry_idx >= len(df) or exit_idx >= len(df):
            continue
        
        entry_price = float(trade['entry_price'])
        exit_price = float(trade.get('exit_price', entry_price))
        direction = trade['direction']
        outcome = trade.get('outcome', 'OPEN')
        
        if direction == "LONG":
            long_entry[entry_idx] = df.iloc[entry_idx]['low'] * 0.998
            if outcome == "WIN":
                long_exit_win[exit_idx] = df.iloc[exit_idx]['high'] * 1.002
            else:
                long_exit_loss[exit_idx] = df.iloc[exit_idx]['high'] * 1.002
        else:
            short_entry[entry_idx] = df.iloc[entry_idx]['high'] * 1.002
            if outcome == "WIN":
                short_exit_win[exit_idx] = df.iloc[exit_idx]['low'] * 0.998
            else:
                short_exit_loss[exit_idx] = df.iloc[exit_idx]['low'] * 0.998
    
    # Add plots
    apds = []
    
    if any(not np.isnan(x) for x in long_entry):
        apds.append(mpf.make_addplot(long_entry, type='scatter', markersize=100, marker='^', color='#00C853'))
    if any(not np.isnan(x) for x in long_exit_win):
        apds.append(mpf.make_addplot(long_exit_win, type='scatter', markersize=100, marker='v', color='#00C853'))
    if any(not np.isnan(x) for x in long_exit_loss):
        apds.append(mpf.make_addplot(long_exit_loss, type='scatter', markersize=100, marker='v', color='#FF1744'))
    if any(not np.isnan(x) for x in short_entry):
        apds.append(mpf.make_addplot(short_entry, type='scatter', markersize=100, marker='v', color='#FF1744'))
    if any(not np.isnan(x) for x in short_exit_win):
        apds.append(mpf.make_addplot(short_exit_win, type='scatter', markersize=100, marker='^', color='#00C853'))
    if any(not np.isnan(x) for x in short_exit_loss):
        apds.append(mpf.make_addplot(short_exit_loss, type='scatter', markersize=100, marker='^', color='#FF1744'))
    
    # Title
    title = f"{symbol} {timeframe} Backtest Overview\n"
    title += f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}% | PnL: {pnl}R"
    
    # Style
    mc = mpf.make_marketcolors(
        up='#00C853', down='#FF1744',
        edge={'up': '#00C853', 'down': '#FF1744'},
        wick={'up': '#00C853', 'down': '#FF1744'},
        volume='#64B5F6'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#333333',
        facecolor='#1a1a2e',
        figcolor='#1a1a2e',
        rc={
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white'
        }
    )
    
    # Plot
    if apds:
        mpf.plot(
            df,
            type='candle',
            style=style,
            title=title,
            ylabel='Price ($)',
            volume=True,
            addplot=apds,
            figsize=(20, 12),
            tight_layout=True
        )
    else:
        mpf.plot(
            df,
            type='candle',
            style=style,
            title=title,
            ylabel='Price ($)',
            volume=True,
            figsize=(20, 12),
            tight_layout=True
        )
    
    plt.show()


def print_trade_summary(trades: list):
    """Trade'lerin özet listesini yazdırır."""
    print("\n" + "="*80)
    print(f"{'#':<4} {'Direction':<8} {'Entry Price':<14} {'Exit Price':<14} {'Outcome':<12} {'PnL R':<10} {'Duration'}")
    print("="*80)
    
    for i, trade in enumerate(trades):
        if trade.get('outcome') is None:
            continue
        
        direction = trade['direction']
        entry = float(trade['entry_price'])
        exit_p = float(trade.get('exit_price', entry))
        outcome = trade.get('outcome', 'OPEN')
        pnl = trade.get('pnl_r', 0)
        duration = trade.get('duration_candles', 0)
        
        # Renk kodları
        if outcome == "WIN":
            color = '\033[92m'  # Yeşil
        elif outcome == "LOSS":
            color = '\033[91m'  # Kırmızı
        elif outcome == "BE":
            color = '\033[93m'  # Sarı
        else:
            color = '\033[0m'
        reset = '\033[0m'
        
        print(f"{i:<4} {direction:<8} ${entry:<13,.2f} ${exit_p:<13,.2f} {color}{outcome:<12}{reset} {pnl:<10} {duration} candles")
    
    print("="*80)


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def interactive_menu():
    """Interaktif menü gösterir."""
    backtest_dir = "backtest_results"
    
    while True:
        print("\n" + "="*60)
        print("    📊 BACKTEST VISUALIZATION TOOL")
        print("="*60)
        
        files = list_backtest_files(backtest_dir)
        
        if not files:
            print("❌ Backtest dosyası bulunamadı!")
            print(f"   '{backtest_dir}' klasöründe *_backtest.json dosyası olmalı.")
            return
        
        print("\n📁 Mevcut Backtest Dosyaları:")
        for i, f in enumerate(files):
            print(f"   [{i+1}] {f}")
        
        print("\n   [0] Çıkış")
        
        try:
            choice = input("\n👉 Dosya seçin (numara): ").strip()
            
            if choice == '0' or choice.lower() == 'q':
                print("\n👋 Görüşürüz!")
                break
            
            idx = int(choice) - 1
            if idx < 0 or idx >= len(files):
                print("❌ Geçersiz seçim!")
                continue
            
            selected_file = files[idx]
            filepath = os.path.join(backtest_dir, f"{selected_file}_backtest.json")
            
            # Load data
            print(f"\n⏳ Yükleniyor: {selected_file}...")
            result = load_backtest_results(filepath)
            
            symbol = result['symbol']
            timeframe = result['timeframe']
            trades = result.get('signals', [])
            
            if not trades:
                print("❌ Bu backtest'te trade bulunamadı!")
                continue
            
            # Fetch candle data
            print(f"⏳ Binance'den {symbol} {timeframe} verisi çekiliyor...")
            df = fetch_binance_candles(symbol, timeframe, limit=1000)
            
            if df.empty:
                print("❌ Candle verisi alınamadı!")
                continue
            
            # Trade menu
            while True:
                print(f"\n" + "-"*50)
                print(f"📈 {symbol} {timeframe} - {len(trades)} trade")
                print("-"*50)
                print("\n   [1] Tüm trade'leri göster (özet chart)")
                print("   [2] Trade listesi")
                print("   [3] Belirli trade'i görselleştir")
                print("   [4] Ardışık trade'leri gez")
                print("   [0] Geri")
                
                sub_choice = input("\n👉 Seçim: ").strip()
                
                if sub_choice == '0':
                    break
                elif sub_choice == '1':
                    visualize_all_trades(trades, df, symbol, timeframe, result)
                elif sub_choice == '2':
                    print_trade_summary(trades)
                elif sub_choice == '3':
                    print_trade_summary(trades)
                    trade_num = input("\n👉 Trade numarası girin: ").strip()
                    try:
                        t_idx = int(trade_num)
                        if 0 <= t_idx < len(trades):
                            visualize_single_trade(trades[t_idx], df, symbol, timeframe, t_idx)
                        else:
                            print("❌ Geçersiz trade numarası!")
                    except ValueError:
                        print("❌ Geçerli bir numara girin!")
                elif sub_choice == '4':
                    for i, trade in enumerate(trades):
                        if trade.get('outcome') is None:
                            continue
                        visualize_single_trade(trade, df, symbol, timeframe, i)
                        cont = input("\n[Enter] Sonraki trade | [q] Çık: ").strip().lower()
                        if cont == 'q':
                            break
                else:
                    print("❌ Geçersiz seçim!")
                    
        except ValueError:
            print("❌ Geçerli bir numara girin!")
        except KeyboardInterrupt:
            print("\n\n👋 Görüşürüz!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest Visualization Tool')
    parser.add_argument('--file', '-f', type=str, help='Backtest dosyası (örn: BTCUSDT_1h)')
    parser.add_argument('--trade', '-t', type=int, help='Görselleştirilecek trade numarası')
    parser.add_argument('--all', '-a', action='store_true', help='Tüm trade\'leri göster')
    
    args = parser.parse_args()
    
    if args.file:
        # Direct file mode
        backtest_dir = "backtest_results"
        filepath = os.path.join(backtest_dir, f"{args.file}_backtest.json")
        
        if not os.path.exists(filepath):
            print(f"❌ Dosya bulunamadı: {filepath}")
            return
        
        result = load_backtest_results(filepath)
        symbol = result['symbol']
        timeframe = result['timeframe']
        trades = result.get('signals', [])
        
        print(f"⏳ Binance'den {symbol} {timeframe} verisi çekiliyor...")
        df = fetch_binance_candles(symbol, timeframe, limit=1000)
        
        if df.empty:
            print("❌ Candle verisi alınamadı!")
            return
        
        if args.trade is not None:
            if 0 <= args.trade < len(trades):
                visualize_single_trade(trades[args.trade], df, symbol, timeframe, args.trade)
            else:
                print(f"❌ Geçersiz trade numarası. 0-{len(trades)-1} arası olmalı.")
        elif args.all:
            visualize_all_trades(trades, df, symbol, timeframe, result)
        else:
            print_trade_summary(trades)
            visualize_all_trades(trades, df, symbol, timeframe, result)
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
