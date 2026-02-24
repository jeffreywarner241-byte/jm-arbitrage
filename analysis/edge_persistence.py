"""
Spread Persistence
====================================================
- Builds a snapshot-level "live" flag using Net_After_Latency and liquidity gating.
- Extracts contiguous live segments per Opportunity_ID ("events").
- Summarizes event durations (median duration = empirical half-life proxy).


Assumptions:
- Net_After_Latency is in $ per contract (e.g., 0.01 = 1¢).
- Buy_VWAP is in $ per contract.
- L1_depth_$ is dollars of notional depth available at L1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass(frozen=True)
class ColMap:
    """Column mapping for adapting to different CSV schemas."""
    opportunity_id: str = "Opportunity_ID"
    timestamp: str = "timestamp"
    net_edge: str = "Net_After_Latency"
    buy_vwap: str = "Buy_VWAP"
    l1_depth: str = "L1_depth_$"


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(map(str, df.columns))
        )


def build_snapshot_state(
    df: pd.DataFrame,
    colmap: ColMap = ColMap(),
    edge_threshold: float = 0.01,  # $ per contract (1¢)
    min_l1_depth: float = 25.0,    # $ notional depth at L1
) -> pd.DataFrame:
    """
    Compute live state per snapshot.

    Returns a DataFrame with standardized columns:
        Opportunity_ID, timestamp, Net_After_Latency, Buy_VWAP, ROI_decimal, L1_depth_$, is_live
    """
    _require_columns(df, [
        colmap.opportunity_id,
        colmap.timestamp,
        colmap.net_edge,
        colmap.buy_vwap,
        colmap.l1_depth,
    ])

    out = df.copy()

    # Parse timestamp robustly
    out[colmap.timestamp] = pd.to_datetime(out[colmap.timestamp], errors="coerce")
    if out[colmap.timestamp].isna().any():
        bad = out[out[colmap.timestamp].isna()].head(5)
        raise ValueError(
            "Some timestamps could not be parsed. Example rows:\n"
            f"{bad[[colmap.opportunity_id, colmap.timestamp]].to_string(index=False)}"
        )

    # ROI is dimensionless: ($/contract) / ($/contract)
    out["ROI_decimal"] = out[colmap.net_edge] / out[colmap.buy_vwap]

    # "Live" = edge above threshold AND liquidity meets minimum
    out["is_live"] = (out[colmap.net_edge] >= edge_threshold) & (out[colmap.l1_depth] >= min_l1_depth)

    # Standardize column names
    out = out.rename(columns={
        colmap.opportunity_id: "Opportunity_ID",
        colmap.timestamp: "timestamp",
        colmap.net_edge: "Net_After_Latency",
        colmap.buy_vwap: "Buy_VWAP",
        colmap.l1_depth: "L1_depth_$",
    })

    return out[[
        "Opportunity_ID",
        "timestamp",
        "Net_After_Latency",
        "Buy_VWAP",
        "ROI_decimal",
        "L1_depth_$",
        "is_live",
    ]]


def extract_persistence_events(df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contiguous live segments ("events") per Opportunity_ID.

    Input df_state must include:
        Opportunity_ID, timestamp (datetime), is_live (bool), Net_After_Latency, ROI_decimal
    """
    _require_columns(df_state, [
        "Opportunity_ID",
        "timestamp",
        "is_live",
        "Net_After_Latency",
        "ROI_decimal",
    ])

    events: list[Dict[str, Any]] = []

    for oid, g in df_state.groupby("Opportunity_ID", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        live = g["is_live"].astype(bool).to_numpy()
        ts = g["timestamp"]

        i = 0
        n = len(g)
        while i < n:
            if not live[i]:
                i += 1
                continue

            start_idx = i
            while i < n and live[i]:
                i += 1
            end_idx = i - 1

            start_ts = ts.iloc[start_idx]
            end_ts = ts.iloc[end_idx]
            duration_seconds = (end_ts - start_ts).total_seconds()

            censored = (end_idx == n - 1)

            segment = g.iloc[start_idx:end_idx + 1]

            events.append({
                "Opportunity_ID": oid,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_seconds": float(duration_seconds),
                "censored": bool(censored),
                "start_Net_After_Latency": float(segment["Net_After_Latency"].iloc[0]),
                "max_Net_After_Latency": float(segment["Net_After_Latency"].max()),
                "start_ROI_decimal": float(segment["ROI_decimal"].iloc[0]),
            })

    return pd.DataFrame(events)


def summarize_persistence(df_events: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a small summary dict. Median uncensored duration is the 'half-life proxy'.
    """
    if df_events.empty:
        return {
            "num_events": 0,
            "censor_rate": None,
            "median_duration_seconds_uncensored": None,
            "mean_duration_seconds_uncensored": None,
        }

    censor_rate = float(df_events["censored"].mean())
    uncensored = df_events[~df_events["censored"]]

    median_unc = float(uncensored["duration_seconds"].median()) if not uncensored.empty else None
    mean_unc = float(uncensored["duration_seconds"].mean()) if not uncensored.empty else None

    return {
        "num_events": int(len(df_events)),
        "censor_rate": censor_rate,
        "median_duration_seconds_uncensored": median_unc,
        "mean_duration_seconds_uncensored": mean_unc,
    }
