# Systematic_Energy_Trading

**Quantitative energy trading lab for rapid strategy prototyping, futures analytics, rolling logic, intelligent pairs trading, and more — designed to uncover edge in energy markets.**

## Overview

This is a work-in-progress Streamlit dashboard and research toolkit for building and evaluating systematic energy trading strategies. It combines Python-based signal modeling with intuitive UI features to accelerate ideation and strategy refinement.

## Key Features

- **Rolling Futures Construction**  
  Build continuous price series using end-of-month or expiry-calendar-based roll mandates.

- **Futures Curve Visualization**  
  Create animated futures curve GIFs to observe term structure dynamics over time.

- **Modular Strategy Engine**  
  Plug-and-play framework to apply:
  - Momentum strategies (simple, time-weighted, crossover, multi-frequency)
  - Carry-based strategies
  - Inventory signals
  - Behavioral & congestion logic

- **Interactive Signal Testing**  
  Visual overlays, quick parameter tuning, and instant feedback across strategy variants.

- **Sharpe-Optimized Strategy Tuning** (Coming Soon)  
  In-dashboard optimization engine to test parameters and maximize performance.

## Status

This is an active project under development. Next milestones:

- Add metrics (Sharpe, drawdowns, cumulative returns)  
- Expand futures coverage across agriculture/metals  
- Integrate backtesting + risk overlays  
- Add pairs/spread selection logic  
- Enable export to research notebooks

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/main.py
