"""
JM Alpha — Spread Persistence Demo

Run from repo root:

    python3 analysis/spread_persistence_demo.py data/raw/your_file.csv

Optional overrides:

    python3 analysis/spread_persistence_demo.py data/raw/your_file.csv \
        --edge_threshold 0.01 \
        --min_l1_depth 25

If your CSV uses different column names:

    python3 analysis/spread_persistence_demo.py data/raw/your_file.csv \
        --opportunity_id Opportunity_ID \
        --timestamp timestamp \
        --net_edge Net_After_Latency \
        --buy_vwap Buy_VWAP \
        --l1_depth L1_depth_$
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

# --- Robust import fix ---
# Ensures repo root is on sys.path even if script is run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.spread_persistence import (
    ColMap,
    build_snapshot_state,
    extract_persistence_events,
    summarize_persistence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JM Alpha spread persistence on a snapshot CSV."
    )

    parser.add_argument("csv_path", type=str,
                        help="Path to snapshot CSV (e.g., data/raw/snapshots.csv)")

    parser.add_argument("--edge_threshold", type=float, default=0.01,
                        help="Edge threshold in $/contract (default 0.01 = 1¢)")

    parser.add_argument("--min_l1_depth", type=float, default=25.0,
                        help="Minimum L1 depth in $ (default 25)")

    # Column overrides
    parser.add_argument("--opportunity_id", type=str, default="Opportunity_ID")
    parser.add_argument("--timestamp", type=str, default="timestamp")
    parser.add_argument("--net_edge", type=str, default="Net_After_Latency")
    parser.add_argument("--buy_vwap", type=str, default="Buy_VWAP")
    parser.add_argument("--l1_depth", type=str, default="L1_depth_$")

    parser.add_argument("--out_dir", type=str, default="data/derived",
                        help="Output directory (default data/derived)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    print(f"\nLoading data from: {csv_path.resolve()}")
    df_raw = pd.read_csv(csv_path)

    colmap = ColMap(
        opportunity_id=args.opportunity_id,
        timestamp=args.timestamp,
        net_edge=args.net_edge,
        buy_vwap=args.buy_vwap,
        l1_depth=args.l1_depth,
    )

    print("Building snapshot state...")
    df_state = build_snapshot_state(
        df_raw,
        colmap=colmap,
        edge_threshold=args.edge_threshold,
        min_l1_depth=args.min_l1_depth,
    )

    print("Extracting persistence events...")
    df_events = extract_persistence_events(df_state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "persistence_events.csv"
    df_events.to_csv(out_path, index=False)

    summary = summarize_persistence(df_events)

    print("\n--- Spread Persistence Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved events to: {out_path.resolve()}")
    print(f"Number of events detected: {len(df_events)}")


if __name__ == "__main__":
    main()
