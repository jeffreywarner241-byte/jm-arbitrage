"""
Execution Realism Module
============================

Demonstrates why quoted arbitrage ≠ realizable arbitrage once you account for:
- Orderbook depth
- Slippage (VWAP degradation)
- Partial fills
- Transaction fees

Author: Jeff Warner | JM Alpha Arbitrage Bot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.precision', 4)


# =============================================================================
# CORE FUNCTION: Walk Orderbook and Calculate VWAP
# =============================================================================

def walk_book_vwap(orderbook, target_qty, side='ask'):
    """
    Walk through orderbook levels and calculate Volume-Weighted Average Price.
    
    Parameters:
    -----------
    orderbook : DataFrame
        Must contain either ['Ask_Price', 'Ask_Qty'] or ['Bid_Price', 'Bid_Qty']
    target_qty : int
        Number of contracts to execute
    side : str
        'ask' for buying (walk up asks), 'bid' for selling (walk down bids)
    
    Returns:
    --------
    dict with keys:
        - vwap: Volume weighted average price
        - filled_qty: Total quantity filled
        - unfilled_qty: Quantity that couldn't be filled
        - notional: Total dollar amount (cost for asks, proceeds for bids)
        - levels_used: Number of price levels consumed
        - fully_filled: Boolean, True if entire order filled
        - fill_rate: Percentage of order filled
    
    Notes:
    ------
    - For 'ask' side: notional represents cost (money spent)
    - For 'bid' side: notional represents proceeds (money received)
    - Does NOT assume full execution — surfaces partial fills explicitly
    """
    
    price_col = 'Ask_Price' if side == 'ask' else 'Bid_Price'
    qty_col = 'Ask_Qty' if side == 'ask' else 'Bid_Qty'
       
    # Defensive sort: enforce correct price priority before walking the book
    # asks: lowest price first; bids: highest price first
    ascending = (side == 'ask')
    orderbook = orderbook.sort_values(price_col, ascending=ascending)

    remaining_qty = target_qty
    notional = 0  # Total dollar amount (cost or proceeds)
    filled_qty = 0
    levels_used = 0
    
    for idx, row in orderbook.iterrows():
        if remaining_qty <= 0:
            break
            
        available_at_level = row[qty_col]
        price_at_level = row[price_col]
        
        # Fill as much as possible at this level
        qty_to_fill = min(remaining_qty, available_at_level)
        
        notional += qty_to_fill * price_at_level
        filled_qty += qty_to_fill
        remaining_qty -= qty_to_fill
        levels_used += 1
    
    # Calculate VWAP
    vwap = notional / filled_qty if filled_qty > 0 else 0
    
    # Calculate unfilled quantity and fill status
    unfilled_qty = target_qty - filled_qty
    fully_filled = (unfilled_qty == 0)
    fill_rate = (filled_qty / target_qty) * 100 if target_qty > 0 else 0
    
    return {
        'vwap': vwap,
        'filled_qty': filled_qty,
        'unfilled_qty': unfilled_qty,
        'notional': notional,
        'levels_used': levels_used,
        'fully_filled': fully_filled,
        'fill_rate': fill_rate
    }


# =============================================================================
# MANUAL SANITY CHECK: Tiny 2 Level Orderbook
# =============================================================================

print("="*70)
print("MANUAL SANITY CHECK: Verifying walk_book_vwap() Logic")
print("="*70)

# Create a trivial orderbook we can hand-calculate
tiny_book = pd.DataFrame({
    'Level': [1, 2],
    'Ask_Price': [0.50, 0.60],
    'Ask_Qty': [100, 200]
})

print("\\nTiny Orderbook (Ask Side):")
print(tiny_book)

# Test case: Buy 150 contracts
test_qty = 150

print(f"\\nTest: Buying {test_qty} contracts")
print("\\nExpected (hand calculation):")
print("  - Fill 100 @ $0.50 = $50")
print("  - Fill 50 @ $0.60 = $30")
print("  - Total cost: $80")
print("  - VWAP: $80 / 150 = $0.5333")

result = walk_book_vwap(tiny_book, test_qty, side='ask')

print(f"\\nActual (from function):")
print(f"  - Notional: ${result['notional']:.2f}")
print(f"  - VWAP: ${result['vwap']:.4f}")
print(f"  - Filled: {result['filled_qty']} contracts")
print(f"  - Unfilled: {result['unfilled_qty']} contracts")
print(f"  - Fully Filled: {result['fully_filled']}")
print(f"  - Fill Rate: {result['fill_rate']:.1f}%")

assert abs(result['vwap'] - 0.5333) < 0.0001, "VWAP calculation error!"
print("\\n✅ Sanity check passed: Function matches hand calculation.")


# =============================================================================
# V1: Single Orderbook Analysis with Partial Fill Handling
# =============================================================================

print("\\n" + "="*70)
print("V1: Single Orderbook Analysis (Buy Side)")
print("="*70)

# Simplified/exaggerated orderbook for demo
orderbook = pd.DataFrame({
    'Level': [1, 2, 3, 4, 5],
    'Ask_Price': [0.50, 0.51, 0.53, 0.56, 0.60],
    'Ask_Qty': [100, 200, 300, 500, 1000]
})

print("\\nMock Orderbook (Ask Side):")
print(orderbook)
print(f"\\nTotal Available Depth: {orderbook['Ask_Qty'].sum()} contracts")

# Test multiple trade sizes
trade_sizes = [100, 200, 500, 1000, 2500]
best_price = orderbook.iloc[0]['Ask_Price']

results = []
for size in trade_sizes:
    result = walk_book_vwap(orderbook, size, side='ask')
    
    slippage_dollars = result['vwap'] - best_price
    slippage_pct = (slippage_dollars / best_price) * 100
    
    results.append({
        'Trade_Size': size,
        'Best_Price': best_price,
        'VWAP': result['vwap'],
        'Slippage_$': slippage_dollars,
        'Slippage_%': slippage_pct,
        'Filled_Qty': result['filled_qty'],
        'Unfilled_Qty': result['unfilled_qty'],
        'Fill_Rate_%': result['fill_rate'],
        'Fully_Filled': result['fully_filled'],
        'Levels_Used': result['levels_used']
    })

results_df = pd.DataFrame(results)

print("\\n=== VWAP Analysis Results ===")
print(results_df.to_string(index=False))

# Check for partial fills
if not results_df['Fully_Filled'].all():
    print("\\n⚠️  WARNING: Some trade sizes exceeded available liquidity.")
    print("    Partial fills occurred. Adjust position size accordingly.")


# =============================================================================
# V2: Bidirectional Support (Buy & Sell)
# =============================================================================

print("\\n" + "="*70)
print("V2: Bidirectional Support (Buy & Sell)")
print("="*70)

full_orderbook = pd.DataFrame({
    'Level': [1, 2, 3, 4, 5],
    'Bid_Price': [0.48, 0.47, 0.45, 0.43, 0.40],
    'Bid_Qty': [100, 200, 300, 500, 1000],
    'Ask_Price': [0.50, 0.51, 0.53, 0.56, 0.60],
    'Ask_Qty': [100, 200, 300, 500, 1000]
})

print("\\nFull Orderbook (Bid + Ask):")
print(full_orderbook)

test_qty = 500

# Buy side
result_buy = walk_book_vwap(full_orderbook, test_qty, side='ask')

# Sell side
result_sell = walk_book_vwap(full_orderbook, test_qty, side='bid')

print(f"\\n=== Bidirectional VWAP Comparison ===")
print(f"\\nBuying {test_qty} contracts (walking asks):")
print(f"  VWAP: ${result_buy['vwap']:.4f}")
print(f"  Notional (cost): ${result_buy['notional']:.2f}")
print(f"  Fill Rate: {result_buy['fill_rate']:.1f}%")

print(f"\\nSelling {test_qty} contracts (walking bids):")
print(f"  VWAP: ${result_sell['vwap']:.4f}")
print(f"  Notional (proceeds): ${result_sell['notional']:.2f}")
print(f"  Fill Rate: {result_sell['fill_rate']:.1f}%")

print(f"\\n📊 Spread (VWAP Buy - VWAP Sell): ${result_buy['vwap'] - result_sell['vwap']:.4f}")


# =============================================================================
# V3: Cross-Platform Edge with Fee Model
# =============================================================================

print("\\n" + "="*70)
print("V3: Cross-Platform Arbitrage with Fees")
print("="*70)

polymarket_book = pd.DataFrame({
    'Level': [1, 2, 3, 4, 5],
    'Ask_Price': [0.48, 0.49, 0.50, 0.52, 0.55],
    'Ask_Qty': [150, 250, 400, 600, 1200]
})

kalshi_book = pd.DataFrame({
    'Level': [1, 2, 3, 4, 5],
    'Bid_Price': [0.53, 0.52, 0.51, 0.50, 0.48],
    'Bid_Qty': [120, 200, 350, 550, 1000]
})

print("Polymarket Orderbook (Ask Side for Buying):")
print(polymarket_book)
print("\\nKalshi Orderbook (Bid Side for Selling):")
print(kalshi_book)

# =============================================================================
# FEE MODEL DISCLAIMER
# =============================================================================
print("\\n" + "-"*70)
print("⚠️  FEE MODEL DISCLAIMER")
print("-"*70)
print("The fee formula below is a SIMPLIFIED PLACEHOLDER.")
print("Replace with venue-specific fee schedules before production use.")
print("\\nCurrent formula: fee_impact = fee_rate × VWAP × (1 - VWAP)")
print("This assumes binary outcome pricing. Adjust for your actual fee structure.")
print("-"*70 + "\\n")

# Parameterized fees (easy to swap)
POLYMARKET_MAKER_FEE = 0.00
KALSHI_FEE_RATE = 0.07
LEG_LAG = 0.035  # Execution delay risk


def calculate_realizable_edge(poly_book, kalshi_book, trade_size, 
                               kalshi_fee=KALSHI_FEE_RATE, leg_lag=LEG_LAG):
    """
    Calculate realizable arbitrage edge after fees and slippage.
    
    Strategy: Buy on Polymarket, Sell on Kalshi
    
    Parameters:
    -----------
    kalshi_fee : float
        Fee rate for Kalshi (parameterized for easy replacement)
    leg_lag : float
        Execution delay / slippage assumption
    """
    
    # Buy on Polymarket
    result_buy = walk_book_vwap(poly_book, trade_size, side='ask')
    
    # Sell on Kalshi
    result_sell = walk_book_vwap(kalshi_book, trade_size, side='bid')

    # Compute executable quantity
    executable_qty = min(result_buy['filled_qty'], result_sell['filled_qty'])

    
    # Check if both legs fully filled
    if not (result_buy['fully_filled'] and result_sell['fully_filled']):
        print(f"⚠️  Partial fill detected for size {trade_size}")
    
    vwap_buy = result_buy['vwap']
    vwap_sell = result_sell['vwap']
    
    # Raw edge
    raw_edge = vwap_sell - vwap_buy
    
    # Fee impact (SIMPLIFIED — replace with actual fee schedule)
    fee_impact = kalshi_fee * vwap_buy * (1 - vwap_buy)
    
    # Slippage/leg lag
    slippage_impact = leg_lag
    
    # Net realizable edge
    net_edge = raw_edge - fee_impact - slippage_impact
    
    return {
        'Trade_Size': trade_size,
        'VWAP_Buy_Poly': vwap_buy,
        'VWAP_Sell_Kalshi': vwap_sell,
        'Raw_Edge': raw_edge,
        'Fee_Impact': fee_impact,
        'Slippage_Impact': slippage_impact,
        'Net_Edge': net_edge,
        'Executable_Qty': executable_qty,
        'Realized_Profit_$': net_edge * executable_qty,
        'Fill_Rate_%': (executable_qty / trade_size) * 100 if trade_size > 0 else 0
    }



arb_results = []
for size in trade_sizes:
    result = calculate_realizable_edge(polymarket_book, kalshi_book, size)
    arb_results.append(result)

arb_df = pd.DataFrame(arb_results)

print("\\n=== Cross-Platform Arbitrage Analysis ===")
print(arb_df.to_string(index=False))

print("\\n🔴 Key Insight: Raw edge degrades significantly after fees and slippage.")
print("   What looks like a 5% arb might be 1-2% realizable (or negative at scale).")


# =============================================================================
# VISUALIZATION: Simple, Clean Plots
# =============================================================================

# Plot 1: Slippage vs Trade Size
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results_df['Trade_Size'], results_df['Slippage_%'], marker='o', linewidth=2)
ax.set_xlabel('Trade Size (contracts)')
ax.set_ylabel('Slippage (%)')
ax.set_title('Slippage Growth as Trade Size Increases')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/slippage_vs_size.png', dpi=150)
print("\\n📊 Plot saved: /tmp/slippage_vs_size.png")

# Plot 2: Edge Degradation
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(arb_df['Trade_Size'], arb_df['Raw_Edge'] * 100, marker='o', label='Raw Edge (%)', linewidth=2)
ax.plot(arb_df['Trade_Size'], arb_df['Net_Edge'] * 100, marker='s', label='Net Realizable Edge (%)', linewidth=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
ax.set_xlabel('Trade Size (contracts)')
ax.set_ylabel('Edge (%)')
ax.set_title('Quoted vs Realizable Arbitrage Edge')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/edge_degradation.png', dpi=150)
print("📊 Plot saved: /tmp/edge_degradation.png")


# =============================================================================
# SUMMARY
# =============================================================================

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Key Learnings:

1. Quoted spreads mislead — Limited depth at best prices.

2. VWAP > Best Price — Walking orderbook levels degrades execution price.

3. Slippage grows non-linearly — Larger trades hit worse prices exponentially.

4. Partial fills happen — Don't assume full execution. Check fill rates.

5. Fees and leg lag matter — A 5% raw edge can become 1-2% net (or negative).

6. Position sizing is critical — Size trades based on realizable edge at depth.


---
Built by: Jeff Warner & Machs Weiscelbaum | JM Alpha
License: MIT
""")

print("✅ Execution Realism Module complete.")
