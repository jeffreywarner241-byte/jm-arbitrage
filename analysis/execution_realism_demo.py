from pathlib import Path
import pandas as pd

from execution_realism_module import (
    walk_book_vwap,
    slippage_vs_size,
    cross_venue_edge,
    save_slippage_plot,
)

OUTDIR = Path("outputs/figures")


def demo_1_sanity_check():
    print("\n" + "=" * 80)
    print("DEMO 1 — Sanity check: 2-level book (hand-verifiable VWAP)")
    print("=" * 80)

    tiny_book = pd.DataFrame({
        "Ask_Price": [0.50, 0.60],
        "Ask_Qty":   [100, 200],
    })

    target_qty = 150
    res = walk_book_vwap(tiny_book, target_qty, side="ask")
    print("Tiny book:")
    print(tiny_book.to_string(index=False))
    print(f"\nBuy {target_qty} contracts:")
    print(f"  Filled: {res.filled_qty:.0f} / {res.target_qty:.0f}  (fill_rate={res.fill_rate*100:.1f}%)")
    print(f"  VWAP:   {res.vwap:.4f}  (expected ~0.5333)")
    print(f"  Cost:   ${res.notional:.2f}")
    print(f"  Levels used: {res.levels_used}")


def demo_2_slippage_curve():
    print("\n" + "=" * 80)
    print("DEMO 2 — Slippage curve on a more realistic multi-level orderbook")
    print("=" * 80)

    # A smoother / more realistic ask ladder
    ask_book = pd.DataFrame({
        "Ask_Price": [0.48, 0.485, 0.49, 0.50, 0.515, 0.53, 0.55, 0.58],
        "Ask_Qty":   [  40,    60,   80,  120,   160,  220,  300,  500],
    })

    trade_sizes = [25, 50, 100, 200, 400, 700, 1100, 1500]
    df = slippage_vs_size(ask_book, trade_sizes=trade_sizes, side="ask")

    print("Ask book:")
    print(ask_book.to_string(index=False))
    print("\nSlippage vs size:")
    print(df.to_string(index=False))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    save_slippage_plot(
        df,
        OUTDIR / "slippage_vs_size.png",
        "Slippage vs Trade Size (Walking Asks)",
    )
    print(f"\nSaved plot: {OUTDIR / 'slippage_vs_size.png'}")


def demo_3_partial_fill_and_cross_venue():
    print("\n" + "=" * 80)
    print("DEMO 3 — Partial fills + cross-venue edge (illustrative)")
    print("=" * 80)

    # BUY book (asks)
    poly_book = pd.DataFrame({
        "Ask_Price": [0.48, 0.49, 0.50, 0.515, 0.53],
        "Ask_Qty":   [  80,  120,  180,   250,  400],
    })

    # SELL book (bids)
    kalshi_book = pd.DataFrame({
        "Bid_Price": [0.54, 0.53, 0.525, 0.515, 0.50],
        "Bid_Qty":   [  60,  110,   160,   240,  420],
    })

    trade_size = 1200  # intentionally large to trigger partial fills on one or both legs

    buy_res = walk_book_vwap(poly_book, trade_size, side="ask")
    sell_res = walk_book_vwap(kalshi_book, trade_size, side="bid")

    print("Polymarket (buy) ask book:")
    print(poly_book.to_string(index=False))
    print("\nKalshi (sell) bid book:")
    print(kalshi_book.to_string(index=False))

    print(f"\nAttempted trade size: {trade_size} contracts")
    print(f"  BUY leg filled:  {buy_res.filled_qty:.0f}/{buy_res.target_qty:.0f}  (fill_rate={buy_res.fill_rate*100:.1f}%)  VWAP={buy_res.vwap:.4f}")
    print(f"  SELL leg filled: {sell_res.filled_qty:.0f}/{sell_res.target_qty:.0f} (fill_rate={sell_res.fill_rate*100:.1f}%)  VWAP={sell_res.vwap:.4f}")

    executable = min(buy_res.filled_qty, sell_res.filled_qty)
    print(f"  Executable qty (min of legs): {executable:.0f} contracts")

    # Cross-venue edge (includes *placeholder* fees and *placeholder* latency buffer)
    edge = cross_venue_edge(
        buy_book=poly_book,
        sell_book=kalshi_book,
        trade_size=trade_size,
        latency_buffer=0.015,  # placeholder
        fee_rate=0.03,         # placeholder (lighter than before)
    )

    print("\nCross-venue edge decomposition (NOTE: fee + latency are placeholders):")
    print(f"  Raw edge:      {edge.raw_edge*100:.2f}¢")
    print(f"  Fee impact:    {edge.fee_impact*100:.2f}¢  (placeholder)")
    print(f"  Latency buff:  {edge.latency_impact*100:.2f}¢  (placeholder)")
    print(f"  Net edge:      {edge.net_edge*100:.2f}¢")
    print(f"  Realized PnL:  ${edge.realized_profit:.2f} (net_edge * executable_qty)")


def main():
    demo_1_sanity_check()
    demo_2_slippage_curve()
    demo_3_partial_fill_and_cross_venue()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("What you should notice:")
    print("1) VWAP rises with size (walking the book).")
    print("2) 'Best price' is only valid for tiny sizes.")
    print("3) Partial fills are real—never assume full execution.")
    print("4) Raw cross-venue edge can disappear after realistic frictions.")
    print("\nOutputs:")
    print(f"- Figures saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
