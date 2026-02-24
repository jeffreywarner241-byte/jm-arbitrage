import pandas as pd

from edge_decomposition import EdgeDecompConfig, decompose_edge_curve

def main():
    print("\n" + "=" * 80)
    print("JM Alpha — Edge Decomposition Demo")
    print("=" * 80)

    # Synthetic books for demonstration (replace with real snapshots later)
    buy_book = pd.DataFrame({
        "Ask_Price": [0.48, 0.49, 0.50, 0.515, 0.53],
        "Ask_Qty":   [  80,  120,  180,   250,  400],
    })

    sell_book = pd.DataFrame({
        "Bid_Price": [0.54, 0.53, 0.525, 0.515, 0.50],
        "Bid_Qty":   [  60,  110,   160,   240,  420],
    })

    sizes = [50, 100, 250, 500, 800, 1200, 1600]

    # Keep placeholders explicit and easy to tweak
    cfg = EdgeDecompConfig(
        fee_rate_placeholder=0.03,
        latency_buffer_placeholder=0.015,
        min_net_edge_cents=0.0,
    )

    df = decompose_edge_curve(buy_book, sell_book, sizes, cfg=cfg)

    print("\nBooks (BUY=asks, SELL=bids):")
    print("\nBUY book:")
    print(buy_book.to_string(index=False))
    print("\nSELL book:")
    print(sell_book.to_string(index=False))

    print("\nDecomposition across sizes (cents/contract):")
    print(df.to_string(index=False))

    # An interpretive summary
    best_row = df.loc[df["net_edge_cents"].idxmax()]
    worst_row = df.loc[df["net_edge_cents"].idxmin()]

    print("\n" + "-" * 80)
    print("Quick read:")
    print(f"- Best net edge occurs near size={best_row['trade_size']:.0f} with net_edge={best_row['net_edge_cents']:.2f}¢")
    print(f"- Worst net edge occurs near size={worst_row['trade_size']:.0f} with net_edge={worst_row['net_edge_cents']:.2f}¢")
    print("- If net edge flips negative as size increases, that’s usually VWAP decay + costs dominating.")
    print("- If executable_qty < trade_size, you hit liquidity limits (partial fill risk).")
    print("- Fees/latency are placeholders here; swap them with venue schedules when ready.")
    print("-" * 80)


if __name__ == "__main__":
    main()
