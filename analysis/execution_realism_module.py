"""
Execution Realism Utilities
=====================================

Purpose
-------
A reusable implementation of "orderbook walking" to compute
VWAP and diagnose execution realism considering:

- Limited depth at the top of book
- Slippage as size increases
- Partial fills 
- Simple cross-venue edge calculation (placeholder fee model)

Disclaimers
---------------------
- This is an analysis module, not production trading logic.
- Fee modeling differs by venue and order type, included fee logic is a placeholder.
- Latency, cancellations, queue priority, and venue microstructure can change results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Iterable, Tuple

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Side = Literal["ask", "bid"]  # ask = buy (walk asks up), bid = sell (walk bids down)

logger = logging.getLogger(__name__)


# ----------------------------
# Results objects (typed)
# ----------------------------

@dataclass(frozen=True)
class VWAPResult:
    side: Side
    target_qty: float
    filled_qty: float
    unfilled_qty: float
    levels_used: int
    notional: float  # USD (cost for asks, proceeds for bids)
    vwap: float      # USD per contract
    fully_filled: bool

    @property
    def fill_rate(self) -> float:
        return 0.0 if self.target_qty <= 0 else (self.filled_qty / self.target_qty)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "target_qty": self.target_qty,
            "filled_qty": self.filled_qty,
            "unfilled_qty": self.unfilled_qty,
            "levels_used": self.levels_used,
            "notional": self.notional,
            "vwap": self.vwap,
            "fully_filled": self.fully_filled,
            "fill_rate": self.fill_rate,
        }


# ----------------------------
# Core: walk the book for VWAP
# ----------------------------

def walk_book_vwap(
    orderbook: pd.DataFrame,
    target_qty: float,
    side: Side,
    *,
    bid_price_col: str = "Bid_Price",
    bid_qty_col: str = "Bid_Qty",
    ask_price_col: str = "Ask_Price",
    ask_qty_col: str = "Ask_Qty",
) -> VWAPResult:
    """
    Walk orderbook and compute VWAP for executing `target_qty` contracts.

    Parameters
    ----------
    orderbook:
        DataFrame containing price/qty levels.
        For side="ask", must contain ask_price_col/ask_qty_col.
        For side="bid", must contain bid_price_col/bid_qty_col.
    target_qty:
        Contracts you want to execute. Non-negative.
    side:
        "ask" for buy (walk lowest asks upward),
        "bid" for sell (walk highest bids downward).

    Returns
    -------
    VWAPResult
    """
    if target_qty < 0:
        raise ValueError("target_qty must be non-negative")

    if orderbook is None or len(orderbook) == 0 or target_qty == 0:
        return VWAPResult(
            side=side,
            target_qty=float(target_qty),
            filled_qty=0.0,
            unfilled_qty=float(target_qty),
            levels_used=0,
            notional=0.0,
            vwap=0.0,
            fully_filled=(target_qty == 0),
        )

    if side == "ask":
        price_col, qty_col = ask_price_col, ask_qty_col
        ascending = True
    elif side == "bid":
        price_col, qty_col = bid_price_col, bid_qty_col
        ascending = False
    else:
        raise ValueError("side must be 'ask' or 'bid'")

    missing = [c for c in (price_col, qty_col) if c not in orderbook.columns]
    if missing:
        raise KeyError(f"orderbook missing required columns: {missing}")

    ob = orderbook[[price_col, qty_col]].copy()
    ob = ob.dropna()
    ob[qty_col] = pd.to_numeric(ob[qty_col], errors="coerce")
    ob[price_col] = pd.to_numeric(ob[price_col], errors="coerce")
    ob = ob.dropna()

    # Filter non-sensical levels
    ob = ob[(ob[qty_col] > 0) & (ob[price_col] >= 0)]
    if len(ob) == 0:
        return VWAPResult(
            side=side,
            target_qty=float(target_qty),
            filled_qty=0.0,
            unfilled_qty=float(target_qty),
            levels_used=0,
            notional=0.0,
            vwap=0.0,
            fully_filled=False,
        )

    ob = ob.sort_values(price_col, ascending=ascending)

    remaining = float(target_qty)
    notional = 0.0
    filled = 0.0
    levels_used = 0

    # Iterate level by level
    for _, row in ob.iterrows():
        if remaining <= 0:
            break

        avail = float(row[qty_col])
        price = float(row[price_col])

        take = min(remaining, avail)
        notional += take * price
        filled += take
        remaining -= take
        levels_used += 1

    vwap = (notional / filled) if filled > 0 else 0.0
    unfilled = float(target_qty) - filled
    fully_filled = (unfilled <= 1e-12)

    return VWAPResult(
        side=side,
        target_qty=float(target_qty),
        filled_qty=filled,
        unfilled_qty=max(0.0, unfilled),
        levels_used=levels_used,
        notional=notional,
        vwap=vwap,
        fully_filled=fully_filled,
    )


# ----------------------------
# Convenience: slippage curve
# ----------------------------

def slippage_vs_size(
    orderbook: pd.DataFrame,
    trade_sizes: Iterable[float],
    side: Side,
    *,
    best_price: Optional[float] = None,
    price_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute VWAP and slippage for multiple sizes.

    Slippage is computed as VWAP - best_price (for asks) or best_price - VWAP (for bids)
    so that "positive slippage" means worse execution.
    """
    trade_sizes = list(trade_sizes)
    if len(trade_sizes) == 0:
        return pd.DataFrame()

    # Determine best price at top of book if not supplied
    if best_price is None:
        if side == "ask":
            price_col = price_col_override or "Ask_Price"
            best_price = float(pd.to_numeric(orderbook[price_col], errors="coerce").dropna().min())
        else:
            price_col = price_col_override or "Bid_Price"
            best_price = float(pd.to_numeric(orderbook[price_col], errors="coerce").dropna().max())

    rows = []
    for size in trade_sizes:
        res = walk_book_vwap(orderbook, float(size), side=side)
        if side == "ask":
            slip = res.vwap - best_price
        else:
            slip = best_price - res.vwap

        slip_pct = 0.0 if best_price == 0 else (slip / best_price)

        rows.append({
            "trade_size": float(size),
            "best_price": float(best_price),
            "vwap": res.vwap,
            "slippage_$": slip,
            "slippage_%": slip_pct * 100.0,
            "filled_qty": res.filled_qty,
            "unfilled_qty": res.unfilled_qty,
            "fill_rate_%": res.fill_rate * 100.0,
            "fully_filled": res.fully_filled,
            "levels_used": res.levels_used,
        })

    return pd.DataFrame(rows)


# ----------------------------
# Cross-venue edge (placeholder)
# ----------------------------

@dataclass(frozen=True)
class CrossVenueEdgeResult:
    trade_size: float
    buy_vwap: float
    sell_vwap: float
    raw_edge: float
    fee_impact: float
    latency_impact: float
    net_edge: float
    executable_qty: float
    realized_profit: float  # net_edge * executable_qty

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def kalshi_fee_placeholder(price: float, fee_rate: float = 0.07) -> float:
    """
    Placeholder: fee = fee_rate * p * (1 - p)
    Units: $/contract if price is $/contract.
    Replace with the actual venue schedule for real use.
    """
    p = float(price)
    return float(fee_rate * p * (1.0 - p))


def cross_venue_edge(
    buy_book: pd.DataFrame,
    sell_book: pd.DataFrame,
    trade_size: float,
    *,
    latency_buffer: float = 0.025,
    fee_rate: float = 0.07,
) -> CrossVenueEdgeResult:
    """
    Stylized cross-venue edge:
    - Buy on buy_book (walk asks)
    - Sell on sell_book (walk bids)
    - Apply placeholder fee + latency buffer

    Analysis utility.
    """
    buy_res = walk_book_vwap(buy_book, trade_size, side="ask")
    sell_res = walk_book_vwap(sell_book, trade_size, side="bid")

    executable_qty = min(buy_res.filled_qty, sell_res.filled_qty)

    buy_vwap = buy_res.vwap
    sell_vwap = sell_res.vwap

    raw_edge = sell_vwap - buy_vwap
    fee_impact = kalshi_fee_placeholder(buy_vwap, fee_rate=fee_rate)
    latency_impact = float(latency_buffer)

    net_edge = raw_edge - fee_impact - latency_impact
    realized_profit = net_edge * executable_qty

    return CrossVenueEdgeResult(
        trade_size=float(trade_size),
        buy_vwap=buy_vwap,
        sell_vwap=sell_vwap,
        raw_edge=raw_edge,
        fee_impact=fee_impact,
        latency_impact=latency_impact,
        net_edge=net_edge,
        executable_qty=float(executable_qty),
        realized_profit=float(realized_profit),
    )


# ----------------------------
# Plot helpers
# ----------------------------

def save_slippage_plot(df: pd.DataFrame, outpath: Path, title: str) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df["trade_size"], df["slippage_%"], marker="o", linewidth=2)
    plt.xlabel("Trade Size (contracts)")
    plt.ylabel("Slippage (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
