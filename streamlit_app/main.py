import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from streamlit_app.utilities.signals import get_signal, strategy_functions
from streamlit_app.data_loader import load_commodity_data
from streamlit_app.utilities.visualization import visualize_signal

# Must be the first Streamlit call
st.set_page_config(page_title="Commodity Strategy Visualizer", layout="wide")

st.title("Commodity Strategy Visualizer")

# Load commodity Excel data
DATA_PATH = os.path.expanduser("~/Desktop/NYU_MFE/Semester_2/1.2_Trading_Energy_Derivatives_MATH_GA_2800/Oil_Project/Systematic_Energy_Trading/streamlit_app/data/NGLs.xlsx")
commodity_dict = load_commodity_data(DATA_PATH)

# Sidebar - user input for commodity selection
commodity = st.sidebar.selectbox("Select Commodity Sheet", list(commodity_dict.keys()))
df = commodity_dict[commodity]

# Sidebar - strategy selection
selected_strategies = st.sidebar.multiselect("Select Strategies", list(strategy_functions.keys()), default=["Basic Momentum"])

# Sidebar - strategy parameters
params = {}
st.sidebar.markdown("Strategy Parameters")

for strategy in selected_strategies:
    st.sidebar.markdown(f"**{strategy}**")
    if strategy == "Basic Momentum" or strategy == "Value":
        params[strategy] = {
            "price_col": df.columns[0],
            "window": st.sidebar.number_input(f"{strategy} - Window (Days)", value=20, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Time-Weighted Momentum":
        params[strategy] = {
            "price_col": df.columns[0],
            "window": st.sidebar.number_input(f"{strategy} - Window (Days)", value=5, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Crossover Momentum":
        fast = st.sidebar.number_input(f"{strategy} - Fast MA (Days)", value=5, min_value=1)
        slow = st.sidebar.number_input(f"{strategy} - Slow MA (Days)", value=20, min_value=fast+1)
        params[strategy] = {
            "price_col": df.columns[0],
            "fast": fast,
            "slow": slow,
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Multi-Frequency Momentum":
        params[strategy] = {
            "price_col": df.columns[0],
            "pairs": [(1, 5), (5, 20), (20, 60)],
            "weights": [0.3, 0.3, 0.4],
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Carry":
        params[strategy] = {
            "front_col": df.columns[0],
            "back_col": df.columns[1],
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.05, step=0.01)
        }
    elif strategy == "Carry Momentum":
        params[strategy] = {
            "front_col": df.columns[0],
            "back_col": df.columns[1],
            "window": st.sidebar.number_input(f"{strategy} - Window (Days)", value=20, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.05, step=0.01)
        }
    elif strategy == "Carry of Carry":
        params[strategy] = {
            "front_col": df.columns[0],
            "back_col": df.columns[1],
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.05, step=0.01)
        }
    elif strategy == "Congestion":
        params[strategy] = {
            "spread_col": df.columns[0],
            "roll_day_1": st.sidebar.number_input(f"{strategy} - Roll Day 1", value=5),
            "roll_day_2": st.sidebar.number_input(f"{strategy} - Roll Day 2", value=9)
        }
    elif strategy == "Inventory":
        params[strategy] = {
            "inventory_col": df.columns[0],
            "k1": st.sidebar.number_input(f"{strategy} - Long MA (k1 Days)", value=60, min_value=1),
            "k2": st.sidebar.number_input(f"{strategy} - Short MA (k2 Days)", value=5, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Inventory Units)", value=1.0, step=0.1)
        }

# --- Display section ---
st.markdown("Strategy Outputs")

for strategy in selected_strategies:
    st.markdown(f"{strategy}")

    signal = get_signal(strategy, df.copy(), **params[strategy])
    price_series = df[params[strategy].get("price_col", df.columns[0])]

    fig = visualize_signal(
        df,
        price_col=params[strategy].get("price_col", df.columns[0]),
        signal_series=signal,
        title=f"{strategy} Signal on {commodity}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("*Metrics placeholder: cumulative return, Sharpe, drawdown...*")
    st.markdown("---")

print(sys.version)