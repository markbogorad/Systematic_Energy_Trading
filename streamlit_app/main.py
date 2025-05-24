import os
import sys
import streamlit as st
import pandas as pd

# Set project root path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# === IMPORT MODULES ===
from streamlit_app.utilities.signals import get_signal, apply_strategy_returns, strategy_functions
from streamlit_app.utilities.data_loader import (
    load_filtered_commodities,
    get_filter_options,
    ensure_database
)
from streamlit_app.utilities.visualization import (
    plot_rolling_futures,
    generate_futures_gif,
    visualize_signal
)
from streamlit_app.utilities.rolling_futures import compute_rolling_futures
from streamlit_app.utilities.metrics import compute_strategy_metrics

# === CACHING UTILITIES ===
@st.cache_data
def get_cached_filter_options(data_path):
    return get_filter_options(data_path)

@st.cache_data
def get_cached_commodities(data_path, internal_tag):
    return load_filtered_commodities(data_path, internal_tag, None, None)

@st.cache_data
def get_cached_rolling(df, transaction_cost):
    return compute_rolling_futures(
        df,
        method="eom",
        transaction_cost=transaction_cost,
        f1_col="F1",
        f2_col="F2",
        date_col="date",
        tenor_col="tenor",
        price_col="px_last"
    )

# === PAGE SETUP ===
st.set_page_config(page_title="Commodity Strategy Visualizer", layout="wide")
st.title("Commodity Strategy Visualizer")

# === HARD-CODED TAG DATABASE MAP ===
TAG_TO_FILE_ID = {
    "basics": "1CmUdyLvKneqLIzFnBO-c20d7oOaZvlzY",
    "cme liquid": "1fyeaWrRC0tnb2kM-Vglq9a8ai-vNv-7p",
    "catherine rec": "1TbOf3L6gVzL0QWon_XHoKZmOx9CK2nYT",
    "liquid ice us": "18vg6zzASt2xXefmOpAZU20UC9XjhiWZl"
}

# === SIDEBAR FILTERS ===
right_sidebar = st.sidebar.container()
right_sidebar.header("Filter Futures")

available_tags = sorted(TAG_TO_FILE_ID.keys(), key=lambda x: (x != "catherine rec", x))
internal_tag = right_sidebar.selectbox("Internal Tag", available_tags, key="internal_tag_select")

# Download and load tag-specific database
file_id = TAG_TO_FILE_ID[internal_tag.lower()]
db_path = f"/tmp/commodities_{internal_tag.lower().replace(' ', '_')}.db"
DATA_PATH = ensure_database(file_path=db_path, file_id=file_id)

# Load filtered tickers for selected tag
commodity_dict = get_cached_commodities(DATA_PATH, internal_tag)
raw_df_all = commodity_dict.get("futures", pd.DataFrame())

if raw_df_all.empty or "bbg_ticker" not in raw_df_all.columns:
    st.warning("No commodities found under this tag.")
    st.stop()

non_empty_tickers = (
    raw_df_all.groupby("bbg_ticker")
    .filter(lambda df: not df.empty and df["px_last"].notna().any())
)

if non_empty_tickers.empty:
    st.warning("No tickers with data found under this tag.")
    st.stop()

available = load_filtered_commodities(DATA_PATH, internal_tag, only_metadata=True)
if available.empty:
    st.warning("No available tickers under this tag.")
    st.stop()

available["display"] = available["bbg_ticker"] + " — " + available["description"].fillna("")
selected_display = right_sidebar.selectbox("Select BBG Ticker", available["display"])
selected_ticker = selected_display.split(" — ")[0]

commodity_dict = load_filtered_commodities(DATA_PATH, internal_tag, ticker_search=selected_ticker)
raw_df_all = commodity_dict.get("futures", pd.DataFrame())
raw_df = raw_df_all[raw_df_all["bbg_ticker"] == selected_ticker].copy()

required_cols = {"date", "tenor", "px_last"}
df_cols = set(col.lower() for col in raw_df.columns)
if not required_cols.issubset(df_cols):
    st.error(f"The 'futures' table must contain: {', '.join(required_cols)}")
    st.write("Available columns:", raw_df.columns.tolist())
    st.stop()

# === LEFT SIDEBAR: STRATEGY INPUT ===
left_sidebar = st.sidebar.container()
transaction_cost = left_sidebar.number_input("Transaction Cost", value=0.002, step=0.001, key="transaction_cost_input")

# === Compute Rolling Futures ===
try:
    df = get_cached_rolling(raw_df, transaction_cost)
except Exception as e:
    st.error(f"Rolling futures computation failed: {e}")
    df = None

# === Plot Rolling Futures and Button ===
if df is not None and "Rolling Futures" in df.columns:
    st.subheader("Rolling Futures Time Series")
    st.plotly_chart(plot_rolling_futures(df), use_container_width=True)

    if st.button("Show Futures Term Structure Over Time", key="futures_term_structure_button"):
        with st.spinner("Generating GIF..."):
            gif_path = generate_futures_gif(
                df.pivot(index="date", columns="tenor", values="px_last").reset_index(),
                sheet_name=raw_df['description'].iloc[0] if "description" in raw_df.columns else selected_ticker,
                save_path="futures_curve.gif"
            )
            st.image(gif_path)

# === Strategy Controls ===
left_sidebar.markdown("---")
core_strategies = [s for s in strategy_functions if s in ["Basic Momentum", "Value", "Carry"]]
selected_strategies = left_sidebar.multiselect("Select Strategies", core_strategies, default=["Basic Momentum"], key="strategy_select")
left_sidebar.markdown("Strategy Parameters")

params = {}
for strategy in selected_strategies:
    left_sidebar.markdown(f"**{strategy}**")
    if strategy in ["Basic Momentum", "Value"]:
        params[strategy] = {
            "price_col": "Rolling Futures",
            "window": left_sidebar.number_input(f"{strategy} - Window (Days)", value=20, min_value=1, key=f"{strategy}_window"),
            "threshold": left_sidebar.number_input(f"{strategy} - Threshold", value=0.01, step=0.01, key=f"{strategy}_threshold")
        }
    elif strategy == "Carry":
        valid_tenors = [col for col in df.columns if col.endswith("m") and col[:-1].isdigit()]
        valid_tenors = sorted(valid_tenors, key=lambda x: int(x[:-1]))
        params[strategy] = {
            "front_col": left_sidebar.selectbox("Carry Front Tenor", valid_tenors, index=0, key="carry_front"),
            "back_col": left_sidebar.selectbox("Carry Back Tenor", valid_tenors, index=3, key="carry_back"),
            "threshold": left_sidebar.number_input(f"{strategy} - Threshold", value=0.05, step=0.01, key="carry_threshold")
        }
        left_sidebar.caption("Default is front = 1m, back = 4m")

# === Strategy Outputs ===
st.markdown("Strategy Outputs")
if df is not None:
    for strategy in selected_strategies:
        st.markdown(f"### {strategy}")

        min_date = df["date"].min().date()
        max_date = df["date"].max().date()

        date_range = st.slider(
            f"{strategy} – Select Time Horizon",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=f"{strategy}_date_slider"
        )

        df_filtered = df[
            (df["date"] >= pd.to_datetime(date_range[0])) & 
            (df["date"] <= pd.to_datetime(date_range[1]))
        ].copy()

        signal = get_signal(strategy, df_filtered.copy(), **params[strategy])
        price_col = "Rolling Futures"
        pnl_series = apply_strategy_returns(df_filtered, signal, price_col=price_col)
        pnl_series.index = df_filtered["date"]

        fig = visualize_signal(pnl_series=pnl_series, title=f"{strategy} Strategy PnL")
        st.plotly_chart(fig, use_container_width=True)

        metrics = compute_strategy_metrics(pnl_series)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total P&L", f"{metrics['Total P&L']}")
        col2.metric("Annualized P&L", f"{metrics['Annualized P&L']}")
        col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Annualized Std Dev", f"{metrics['Annualized Std Dev']}")
        col5.metric("High Water Mark", f"{metrics['High Water Mark']}")
        col6.metric("Max Drawdown", f"{metrics['Max Drawdown']}")

        st.metric("Return on Drawdown", f"{metrics['Return on Drawdown']}")

