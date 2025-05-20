import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from streamlit_app.utilities.signals import get_signal, strategy_functions
from streamlit_app.data_loader import load_commodity_data
from streamlit_app.utilities.visualization import plot_rolling_futures, generate_futures_gif, visualize_signal
from streamlit_app.utilities.rolling_futures import compute_rolling_futures

# === PATH CONFIGURATION ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === PAGE SETUP ===
st.set_page_config(page_title="Commodity Strategy Visualizer", layout="wide")
st.title("Commodity Strategy Visualizer")

# === DATA LOADING ===
DATA_PATH = os.path.expanduser("~/Desktop/NYU_MFE/Semester_2/1.2_Trading_Energy_Derivatives_MATH_GA_2800/Oil_Project/Systematic_Energy_Trading/streamlit_app/data/NGLs.xlsx")
commodity_dict = load_commodity_data(DATA_PATH)

# === SIDEBAR: Commodity Selection ===
commodity = st.sidebar.selectbox("Select Commodity Sheet", list(commodity_dict.keys()))
df = commodity_dict[commodity]

# Enforce date column
if df.index.name is not None:
    df = df.reset_index().rename(columns={df.index.name: "Date"})

# === SIDEBAR: Rolling Futures Settings ===
roll_method_display = st.sidebar.selectbox("Rolling Method", ["End of Month", "Roll Mandate + Calendar (coming soon)"])
roll_method_map = {
    "end of month": "eom",
    "roll mandate + calendar": "calendar"
}
roll_method = roll_method_map.get(roll_method_display.lower().strip(), "eom")
transaction_cost = st.sidebar.number_input("Transaction Cost", value=0.002, step=0.001)

calendar_path = None
if roll_method == "calendar":
    calendar_upload = st.sidebar.file_uploader("Upload Expiry Calendar (.xlsx)", type=["xlsx"])
    if calendar_upload is not None:
        calendar_path = calendar_upload
    else:
        st.warning("Please upload an expiry calendar to use calendar-based rolling.")

# === COMPUTE ROLLING FUTURES ===
try:
    df = compute_rolling_futures(
        df,
        method=roll_method,
        transaction_cost=transaction_cost,
        calendar_path=calendar_path,
        f1_col=df.columns[1],
        f2_col=df.columns[2]
    )
except Exception as e:
    st.error(f"Rolling futures computation failed: {e}")
    if "Rolling Futures" not in df.columns:
        st.warning("Falling back to front-month futures as Rolling Futures.")
        df["Rolling Futures"] = df[df.columns[0]]

# === DISPLAY: Rolling Futures Plot ===
if "Rolling Futures" in df.columns:
    st.subheader("Rolling Futures Time Series")
    fig_roll = plot_rolling_futures(df)
    st.plotly_chart(fig_roll, use_container_width=True)

# === SIDEBAR: Strategy Controls ===
st.sidebar.markdown("---")
selected_strategies = st.sidebar.multiselect("Select Strategies", list(strategy_functions.keys()), default=["Basic Momentum"])

params = {}
st.sidebar.markdown("Strategy Parameters")

for strategy in selected_strategies:
    st.sidebar.markdown(f"**{strategy}**")
    if strategy in ["Basic Momentum", "Value"]:
        params[strategy] = {
            "price_col": "Rolling Futures",
            "window": st.sidebar.number_input(f"{strategy} - Window (Days)", value=20, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Time-Weighted Momentum":
        params[strategy] = {
            "price_col": "Rolling Futures",
            "window": st.sidebar.number_input(f"{strategy} - Window (Days)", value=5, min_value=1),
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Crossover Momentum":
        fast = st.sidebar.number_input(f"{strategy} - Fast MA (Days)", value=5, min_value=1)
        slow = st.sidebar.number_input(f"{strategy} - Slow MA (Days)", value=20, min_value=fast+1)
        params[strategy] = {
            "price_col": "Rolling Futures",
            "fast": fast,
            "slow": slow,
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy == "Multi-Frequency Momentum":
        params[strategy] = {
            "price_col": "Rolling Futures",
            "pairs": [(1, 5), (5, 20), (20, 60)],
            "weights": [0.3, 0.3, 0.4],
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.1, step=0.01)
        }
    elif strategy in ["Carry", "Carry Momentum", "Carry of Carry"]:
        params[strategy] = {
            "front_col": df.columns[0],
            "back_col": df.columns[1],
            "threshold": st.sidebar.number_input(f"{strategy} - Threshold (Commodity Units)", value=0.05, step=0.01)
        }
        if strategy == "Carry Momentum":
            params[strategy]["window"] = st.sidebar.number_input(f"{strategy} - Window (Days)", value=20, min_value=1)
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
    elif strategy == "Follow-the-Flow":
        params[strategy] = {
            "mm_col": df.columns[0],
            "m": st.sidebar.number_input(f"{strategy} - Short MA (m)", value=5, min_value=1),
            "n": st.sidebar.number_input(f"{strategy} - Long MA (n)", value=15, min_value=2)
        }
    elif strategy == "Fade-the-Crowded-Trade":
        params[strategy] = {
            "mm_col": df.columns[0],
            "epsilon": st.sidebar.number_input(f"{strategy} - Epsilon", value=0.2, step=0.01, min_value=0.01, max_value=0.49),
            "n": st.sidebar.number_input(f"{strategy} - Lookback Window (n)", value=20, min_value=2)
        }

# === SIDEBAR: Tools ===
st.sidebar.markdown("---")
st.sidebar.subheader("Tools")
generate_gif = st.sidebar.button("🎞️ Generate Futures Curve GIF")
optimize = st.sidebar.button("📈 Optimize Parameters by Sharpe")

# === DISPLAY: GIF GENERATION ===
if generate_gif and "Date" in df.columns:
    st.subheader("Futures Curve Animation")
    df_gif = df.copy().rename(columns={"Date": "date"})
    df_gif["date"] = pd.to_datetime(df_gif["date"])
    futures_cols = [col for col in df_gif.columns if col.lower().startswith("pgp") or "comdty" in col.lower()]
    gif_input_df = df_gif[["date"] + futures_cols]

    with st.spinner("Generating animation... this may take a minute"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                gif_path = generate_futures_gif(
                    gif_input_df,
                    sheet_name=commodity,
                    save_path=tmpfile.name,
                    duration=1000,
                    max_frames=300
                )
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption="Animated Futures Curve", use_column_width=True)
        except Exception as e:
            st.warning(f"GIF generation failed: {e}")

# === DISPLAY: Strategy Results ===
st.markdown("Strategy Outputs")
for strategy in selected_strategies:
    st.markdown(f"### {strategy}")
    signal = get_signal(strategy, df.copy(), **params[strategy])
    fig = visualize_signal(
        df,
        price_col=params[strategy].get("price_col", df.columns[0]),
        signal_series=signal,
        title=f"{strategy} Signal on {commodity}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("*Metrics placeholder: cumulative return, Sharpe, drawdown...*")
    st.markdown("---")
