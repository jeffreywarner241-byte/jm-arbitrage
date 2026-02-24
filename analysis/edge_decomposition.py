"""
JM Alpha — Edge Decomposition (Snapshot)
==============================================

Purpose
-------
Decompose a quoted cross-venue spread into a size-aware, execution-aware "net edge":

Raw edge (Sell VWAP - Buy VWAP)
  - placeholder fees
  - placeholder latency buffer
  = net edge

This module is intended for analysis and portfolio demonstration. Fees/latency are intentionally
parameterized and labeled placeholders.

Units
-----
- Prices are USD per contract (0 to 1 for binary markets)
- Quantities are contracts
- Edge is USD per contract (cents for reporting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Literal

import numpy as np
import pandas as pd

from execution_realism_module import walk_book_vwap

Side = Literal["ask", "bid"]


# ----------------------------
# Config + results
# ----------------------------

@dataclass(frozen=True)
class EdgeDecompConfig:
    """
    Parameters are intentionally simple and explicit.
    Replace these with venue-specific logic once you have verified schedules.
    """
    fee_rate_placeholder: float = 0.03        # placeholder
    latency_buffer_placeholder: float = 0.015 # placeholder ($/contract)
    min_net_edge_cents: float = 0.0           # optional filter threshold


@dataclass(frozen=True)
class EdgeDecompositionResult:
    trade_size: float

    buy_vwap: float
    sell_vwap: float

    buy_filled_qty: float
    sell_filled_qty: float
    executable_qty: float

    raw_edge: float           # $/contract
    fee_impact: float         # $/contract (placeholder)
    latency_impact: float     # $/contract (placeholder)
    net_edge: float           # $/contract

    expected_pnl: float       # (net_edge * executable_qty)

    partial_fill_buy: bool
    partial_fill_sell: bool

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        # Convenience: cents columns for pretty tables
        d["raw_edge_cents"] = 100.0 * d["raw_edge"]
        d["fee_impact_cents"] = 100.0 * d["fee_impact"]
        d["latency_impact_cents"] = 100.0 * d["latency_impact"]
        d["net_edge_cents"] = 100.0 * d["net_edge"]
        return d


# ----------------------------
# Placeholder fee model
# ----------------------------

def binary_fee_placeholder(price: float, fee_rate: float) -> float:
    """
    Placeholder fee model: fee_rate * p * (1-p)

    Returns fee in $/contract.

    Notes
    -----
    - This is NOT a verified venue schedule. But mirrors Kalshi's recent structure.
    - It's only meant to demonstrate that fees are nonlinear in binary prices.
    """
    p = float(price)
    return float(fee_rate * p * (1.0 - p))


# ----------------------------
# Core decomposition
# ----------------------------

def decompose_edge(
    buy_book: pd.DataFrame,
    sell_book: pd.DataFrame,
    trade_size: float,
    *,
    cfg: Optional[EdgeDecompConfig] = None,
) -> EdgeDecompositionResult:
    """
    Decompose edge for a single trade_size.

    Convention
    ----------
    - buy_book is walked on asks (side="ask") -> buy_vwap
    - sell_book is walked on bids (side="bid") -> sell_vwap
    """
    if cfg is None:
        cfg = EdgeDecompConfig()

    trade_size = float(trade_size)

    buy = walk_book_vwap(buy_book, trade_size, side="ask")
    sell = walk_book_vwap(sell_book, trade_size, side="bid")

    executable_qty = float(min(buy.filled_qty, sell.filled_qty))

    buy_vwap = float(buy.vwap)
    sell_vwap = float(sell.vwap)

    raw_edge = sell_vwap - buy_vwap

    # Placeholder costs (explicitly labeled)
    fee_impact = binary_fee_placeholder(buy_vwap, fee_rate=cfg.fee_rate_placeholder)
    latency_impact = float(cfg.latency_buffer_placeholder)

    net_edge = raw_edge - fee_impact - latency_impact
    expected_pnl = net_edge * executable_qty

    partial_fill_buy = (not buy.fully_filled)
    partial_fill_sell = (not sell.fully_filled)

    return EdgeDecompositionResult(
        trade_size=trade_size,
        buy_vwap=buy_vwap,
        sell_vwap=sell_vwap,
        buy_filled_qty=float(buy.filled_qty),
        sell_filled_qty=float(sell.filled_qty),
        executable_qty=executable_qty,
        raw_edge=float(raw_edge),
        fee_impact=float(fee_impact),
        latency_impact=float(latency_impact),
        net_edge=float(net_edge),
        expected_pnl=float(expected_pnl),
        partial_fill_buy=partial_fill_buy,
        partial_fill_sell=partial_fill_sell,
    )


def decompose_edge_curve(
    buy_book: pd.DataFrame,
    sell_book: pd.DataFrame,
    trade_sizes: Iterable[float],
    *,
    cfg: Optional[EdgeDecompConfig] = None,
) -> pd.DataFrame:
    """
    Evaluate decomposition across many sizes; returns a DataFrame for printing or plotting.
    """
    rows = []
    for sz in trade_sizes:
        res = decompose_edge(buy_book, sell_book, float(sz), cfg=cfg)
        rows.append(res.to_dict())

    df = pd.DataFrame(rows)

    # Add a simple "passes_threshold" boolean (optional)
    if cfg is None:
        cfg = EdgeDecompConfig()
    df["passes_min_net_edge"] = df["net_edge_cents"] >= float(cfg.min_net_edge_cents)

    # Pretty ordering
    ordered_cols = [
        "trade_size",
        "executable_qty",
        "buy_vwap",
        "sell_vwap",
        "raw_edge_cents",
        "fee_impact_cents",
        "latency_impact_cents",
        "net_edge_cents",
        "expected_pnl",
        "partial_fill_buy",
        "partial_fill_sell",
        "passes_min_net_edge",
    ]
    cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
    return df[cols]
