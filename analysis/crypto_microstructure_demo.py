"""
Portability Demo (Crypto Microstructure)
==================================================
Goal:
Demonstrate that the execution realism abstraction (orderbook walking / VWAP / partial fills)
generalizes from prediction markets to a continuous-price L2 crypto-style orderbook.

- Uses continuous prices (e.g., BTC-USD)
- Reports spread + slippage in basis points (bps)
- Computes mid-price and compares VWAP to top-of-book and mid
- Sweeps trade sizes in base units and reports notional impact

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt


Level = Tuple[float, float]  # (price, size) where size is in base units (e.g., BTC)


@dataclass
class FillResult:
    side: str
    requested_qty: float
    filled_qty: float
    vwap: float | None
    notional: float
    levels_consumed: int
    partial_fill: bool


def walk_book(levels: List[Level], qty: float) -> FillResult:
    """
    Generic orderbook walking (VWAP) for one side of book.

    Args:
        levels: list of (price, size) in correct consumption order:
            - BUY uses asks sorted ASC by price (best ask first)
            - SELL uses bids sorted DESC by price (best bid first)
        qty: desired base quantity to trade (e.g., BTC)

    Returns:
        FillResult-like dict items (VWAP None if nothing filled)
    """
    remaining = float(qty)
    filled = 0.0
    notional = 0.0
    levels_used = 0

    for price, size in levels:
        if remaining <= 0:
            break
        take = min(remaining, float(size))
        if take > 0:
            filled += take
            notional += take * float(price)
            remaining -= take
            levels_used += 1

    vwap = (notional / filled) if filled > 0 else None
    partial = filled < qty

    return {
        "filled_qty": filled,
        "notional": notional,
        "vwap": vwap,
        "levels_consumed": levels_used,
        "partial_fill": partial,
    }


def bps(diff: float, ref: float) -> float:
    """Convert a price difference to basis points vs ref."""
    return (diff / ref) * 10_000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="crypto microstructure portability demo.")
    p.add_argument("--plot", action="store_true", help="Plot slippage vs size (matplotlib).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Synthetic crypto-style L2 orderbook snapshot ---
    # Prices in USD, sizes in BTC (base units).
    orderbook: Dict[str, Any] = {
        "symbol": "BTC-USD",
        "timestamp": "2026-02-24T16:40:00Z",
        "bids": [  # DESC by price
            (9999.50, 0.20),
            (9999.00, 0.35),
            (9998.50, 0.60),
            (9998.00, 0.80),
            (9997.50, 1.20),
            (9997.00, 1.80),
            (9996.50, 2.50),
        ],
        "asks": [  # ASC by price
            (10000.00, 0.15),
            (10000.50, 0.30),
            (10001.00, 0.55),
            (10001.50, 0.90),
            (10002.00, 1.40),
            (10003.00, 2.00),
            (10004.00, 2.80),
        ],
    }

    bids: List[Level] = orderbook["bids"]
    asks: List[Level] = orderbook["asks"]

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_abs = best_ask - best_bid
    spread_bps = bps(spread_abs, mid)

    print("\n=== JM Alpha — Portability Demo (Crypto Microstructure) ===")
    print(f"Symbol: {orderbook['symbol']}")
    print(f"Timestamp: {orderbook['timestamp']}")
    print(f"Best Bid: {best_bid:.2f} | Best Ask: {best_ask:.2f} | Mid: {mid:.2f}")
    print(f"Spread: {spread_abs:.2f} USD ({spread_bps:.2f} bps)\n")

    # Sweep trade sizes in base units (BTC)
    trade_sizes = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00]

    rows = []
    buy_slippage_bps = []
    sell_slippage_bps = []
    sizes_for_plot = []

    for qty in trade_sizes:
        buy = walk_book(asks, qty)   # buying consumes asks upward
        sell = walk_book(bids, qty)  # selling consumes bids downward

        # Slippage relative to top-of-book
        # BUY adverse slippage = VWAP - best_ask
        # SELL adverse slippage = best_bid - VWAP
        buy_vwap = buy["vwap"]
        sell_vwap = sell["vwap"]

        buy_slip = None if buy_vwap is None else bps(buy_vwap - best_ask, best_ask)
        sell_slip = None if sell_vwap is None else bps(best_bid - sell_vwap, best_bid)

        # VWAP vs mid (also useful in crypto)
        buy_vs_mid = None if buy_vwap is None else bps(buy_vwap - mid, mid)
        sell_vs_mid = None if sell_vwap is None else bps(mid - sell_vwap, mid)

        rows.append({
            "qty": qty,
            "buy_filled": buy["filled_qty"],
            "buy_vwap": buy_vwap,
            "buy_notional": buy["notional"],
            "buy_levels": buy["levels_consumed"],
            "buy_partial": buy["partial_fill"],
            "buy_slip_bps": buy_slip,
            "buy_vs_mid_bps": buy_vs_mid,
            "sell_filled": sell["filled_qty"],
            "sell_vwap": sell_vwap,
            "sell_notional": sell["notional"],
            "sell_levels": sell["levels_consumed"],
            "sell_partial": sell["partial_fill"],
            "sell_slip_bps": sell_slip,
            "sell_vs_mid_bps": sell_vs_mid,
        })

        if buy_slip is not None and sell_slip is not None:
            sizes_for_plot.append(qty)
            buy_slippage_bps.append(buy_slip)
            sell_slippage_bps.append(sell_slip)

    # Pretty print a compact table
    print("Qty(BTC) | Buy VWAP | Buy Slip(bps) | Buy Levels | Partial | Sell VWAP | Sell Slip(bps) | Sell Levels | Partial")
    print("-" * 112)
    for r in rows:
        print(
            f"{r['qty']:7.2f} | "
            f"{(r['buy_vwap'] if r['buy_vwap'] is not None else float('nan')):7.2f} | "
            f"{(r['buy_slip_bps'] if r['buy_slip_bps'] is not None else float('nan')):12.2f} | "
            f"{r['buy_levels']:9d} | "
            f"{str(r['buy_partial']):7s} | "
            f"{(r['sell_vwap'] if r['sell_vwap'] is not None else float('nan')):8.2f} | "
            f"{(r['sell_slip_bps'] if r['sell_slip_bps'] is not None else float('nan')):13.2f} | "
            f"{r['sell_levels']:10d} | "
            f"{str(r['sell_partial']):7s}"
        )

    # Optional plot
    if args.plot and sizes_for_plot:
        plt.figure()
        plt.plot(sizes_for_plot, buy_slippage_bps, marker="o", label="Buy slippage (bps)")
        plt.plot(sizes_for_plot, sell_slippage_bps, marker="o", label="Sell slippage (bps)")
        plt.xlabel("Trade size (BTC)")
        plt.ylabel("Slippage (bps)")
        plt.title("Slippage vs Size — Synthetic Crypto L2 Orderbook")
        plt.legend()
        plt.show()

    print("\nNotes:")
    print("- This uses a synthetic continuous-price orderbook to show portability of VWAP walking.")
    print("- Slippage is reported in bps (crypto-native) vs top-of-book reference prices.")
    print("- Partial fills occur when requested size exceeds available depth on that side.\n")


if __name__ == "__main__":
    main()
