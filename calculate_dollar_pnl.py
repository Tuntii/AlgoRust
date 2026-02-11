"""
Calculate dollar P&L from R-multiple backtest results.

This script converts backtest results from R (risk units) to dollar amounts
using different position sizing strategies.
"""

import json
from decimal import Decimal

def calculate_fixed_fractional(pnl_r: Decimal, starting_capital: float, risk_percent: float) -> dict:
    """
    Calculate P&L using fixed fractional position sizing.
    
    Args:
        pnl_r: Total P&L in R units
        starting_capital: Starting capital in dollars
        risk_percent: Risk per trade as percentage (e.g., 1.0 for 1%)
    
    Returns:
        Dictionary with results
    """
    risk_amount = starting_capital * (risk_percent / 100)  # Dollar amount per 1R
    dollar_pnl = float(pnl_r) * risk_amount
    final_capital = starting_capital + dollar_pnl
    return_pct = (dollar_pnl / starting_capital) * 100
    
    return {
        "starting_capital": starting_capital,
        "risk_per_trade": risk_amount,
        "risk_percent": risk_percent,
        "pnl_r": float(pnl_r),
        "dollar_pnl": round(dollar_pnl, 2),
        "final_capital": round(final_capital, 2),
        "return_percent": round(return_pct, 2)
    }

def calculate_compound(trades_data: list, starting_capital: float, risk_percent: float) -> dict:
    """
    Calculate P&L using compound position sizing (adjusting risk per trade based on current balance).
    
    Note: This requires individual trade data, which we'll need to extract from detailed backtest files.
    """
    # This would require reading individual trades from the detailed JSON files
    # For now, we'll return None and implement if needed
    return None

def main():
    # Load backtest summary
    with open('backtest_results/BACKTEST_SUMMARY.json', 'r') as f:
        summary = json.load(f)
    
    starting_capital_per_pair = 1000.0
    risk_percentages = [1.0, 2.0, 3.0]  # Test different risk levels
    
    print("=" * 80)
    print("💰 DOLLAR P&L CALCULATION FROM BACKTEST RESULTS")
    print("=" * 80)
    print()
    
    # Calculate for each pair
    all_results = []
    
    for result in summary['results_by_pair']:
        pair_name = f"{result['symbol']} {result['timeframe']}"
        pnl_r = Decimal(result['pnl_r'])
        
        print(f"\n📊 {pair_name}")
        print("-" * 80)
        print(f"   Trades: {result['trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   P&L (R): {float(pnl_r):.2f}R")
        print(f"   Expectancy: {result['expectancy']:.4f}R per trade")
        print()
        
        for risk_pct in risk_percentages:
            calc = calculate_fixed_fractional(pnl_r, starting_capital_per_pair, risk_pct)
            all_results.append({
                'pair': pair_name,
                **calc
            })
            
            print(f"   Risk {risk_pct}% per trade (${calc['risk_per_trade']:.2f} per R):")
            print(f"      Starting: ${calc['starting_capital']:,.2f}")
            print(f"      P&L: ${calc['dollar_pnl']:,.2f}")
            print(f"      Final: ${calc['final_capital']:,.2f}")
            print(f"      Return: {calc['return_percent']:,.2f}%")
            print()
    
    # Calculate total across all pairs
    print("\n" + "=" * 80)
    print("🎯 PORTFOLIO SUMMARY (All Pairs Combined)")
    print("=" * 80)
    print()
    
    total_starting = starting_capital_per_pair * len(summary['results_by_pair'])
    overall_pnl_r = Decimal(summary['overall_pnl_r'])
    
    print(f"Number of Pairs: {len(summary['results_by_pair'])}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Overall Win Rate: {summary['overall_win_rate']:.1f}%")
    print(f"Overall P&L: {float(overall_pnl_r):.2f}R")
    print(f"Overall Expectancy: {summary['overall_expectancy']:.4f}R per trade")
    print()
    
    for risk_pct in risk_percentages:
        risk_per_pair = starting_capital_per_pair * (risk_pct / 100)
        
        # Calculate total P&L across all pairs
        total_pnl = 0
        total_final = 0
        
        for result in summary['results_by_pair']:
            pnl_r = Decimal(result['pnl_r'])
            pair_pnl = float(pnl_r) * risk_per_pair
            total_pnl += pair_pnl
            total_final += starting_capital_per_pair + pair_pnl
        
        total_return_pct = (total_pnl / total_starting) * 100
        
        print(f"Risk {risk_pct}% per trade:")
        print(f"   Total Starting Capital: ${total_starting:,.2f}")
        print(f"   Risk per R (per pair): ${risk_per_pair:.2f}")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        print(f"   Total Final Capital: ${total_final:,.2f}")
        print(f"   Total Return: {total_return_pct:.2f}%")
        print()
    
    # Best/Worst performers
    print("=" * 80)
    print(f"🏆 Best Performer: {summary['best_performer']}")
    print(f"⚠️  Worst Performer: {summary['worst_performer']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
