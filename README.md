## Overview

This project is structured as a research framework for analyzing cross-platform
prediction market inefficiencies.

Core analytical components:

- **Execution Realism Module**
  - Models VWAP degradation
  - Quantifies slippage vs size
  - Detects partial fill risk

- **Edge Decomposition Engine**
  - Breaks raw spreads into:
    - VWAP-adjusted edge
    - Fee-adjusted edge
    - Latency-adjusted edge
  - Computes executable quantity
  - Evaluates scalability of opportunities

## Running Analytical Demos

Execution realism:
python analysis/execution_realism_demo.py

Edge decomposition:
python analysis/edge_decomposition_demo.py
