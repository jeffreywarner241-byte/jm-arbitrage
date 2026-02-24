from pathlib import Path
import pandas as pd

from execution_realism_module import slippage_vs_size, cross_venue_edge, save_slippage_plot

def main():
    # Small sanity-check orderbook
    tiny_book = pd.DataFrame({
        "Ask_Price": [0.50, 0.60],
        "Ask_Qty": [100, 200],
    })

    df = slippage_vs_size(tiny_book, trade_sizes=[50, 100, 150, 250], side="ask")
    print(df)

    outdir = Path("outputs/figures")
    save_slippage_plot(df, outdir / "slippage_vs_size_demo.png", "Slippage vs Size (Demo)")

    # Cross venue example
    poly_book = pd.DataFrame({
        "Ask_Price": [0.48, 0.49, 0.50],
        "Ask_Qty": [150, 250, 400],
    })
    kalshi_book = pd.DataFrame({
        "Bid_Price": [0.53, 0.52, 0.51],
        "Bid_Qty": [120, 200, 350],
    })

    edge = cross_venue_edge(poly_book, kalshi_book, trade_size=300)
    print(edge.to_dict())

if __name__ == "__main__":
    main()
