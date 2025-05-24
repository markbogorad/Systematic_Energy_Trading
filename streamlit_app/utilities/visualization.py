import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import matplotlib.patches as mpatches
import pandas as pd

def plot_rolling_futures(df, price_col="Rolling Futures"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[price_col],
        mode="lines",
        name="Rolling Futures"
    ))
    fig.update_layout(
        title="Rolling Futures Time Series",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400
    )
    return fig

def generate_futures_gif(df, sheet_name="Commodity", save_path="futures_curve.gif", duration=1000, max_frames=300):
    """
    Generates an animated GIF of futures curves over time.

    Accepts both long-form (date, tenor, px_last) and wide-form (date + tenor columns) data.
    """

    # Map original column names to lowercase for matching
    col_map = {col.lower(): col for col in df.columns}

    # Detect and convert long-form to wide-form
    if all(key in col_map for key in ["date", "tenor", "px_last"]):
        date_col = col_map["date"]
        tenor_col = col_map["tenor"]
        price_col = col_map["px_last"]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.pivot(index=date_col, columns=tenor_col, values=price_col).reset_index()
        df = df.sort_values(by=date_col)

    elif "date" not in col_map:
        raise KeyError("DataFrame must contain a 'date' column or match the long-form ['date', 'tenor', 'px_last'] structure.")

    # Ensure column names are consistent
    df.columns = [col.lower() for col in df.columns]
    date_col = "date"

    # Extract tenor columns only (e.g., "1m", "2m", ...)
    tenor_cols = [col for col in df.columns if col != "date" and col.endswith("m") and col[:-1].isdigit()]
    tenor_cols = sorted(tenor_cols, key=lambda x: int(x[:-1]))

    if not tenor_cols:
        raise ValueError("No valid tenor columns found. Expected names like '1m', '3m', ...")

    image_folder = "gif_frames"
    os.makedirs(image_folder, exist_ok=True)

    frames = []
    total_rows = len(df)
    step = max(1, total_rows // max_frames)
    selected_rows = df.iloc[::step].reset_index(drop=True)

    all_contracts_patch = mpatches.Patch(color='blue', label='All Contracts')
    winter_patch = mpatches.Patch(color='red', label='Winter Contracts')

    for i, row in selected_rows.iterrows():
        start_month = row[date_col]
        contract_months = [start_month + pd.DateOffset(months=int(col[:-1])) for col in tenor_cols]
        contract_labels = [month.strftime("%Y-%m") for month in contract_months]

        try:
            prices = row[tenor_cols].astype(float).values
        except ValueError:
            continue  # skip frames with bad data

        plt.figure(figsize=(12, 6))
        plt.plot(contract_labels, prices, marker='o', linestyle='-', color='blue')

        for idx, month in enumerate(contract_months):
            if month.month in [12, 1, 2]:
                plt.scatter(contract_labels[idx], prices[idx], color='red', s=100)

        plt.xlabel("Future Contract Month")
        plt.ylabel("Price")
        plt.title(f"Futures Strip Over Time ({sheet_name})\nDate: {start_month.strftime('%Y-%m-%d')}")
        plt.xticks(rotation=45)
        plt.ylim(df[tenor_cols].min().min() - 1, df[tenor_cols].max().max() + 1)
        plt.legend(handles=[all_contracts_patch, winter_patch])

        frame_path = os.path.join(image_folder, f"frame_{i:03d}.png")
        plt.savefig(frame_path, bbox_inches="tight")
        plt.close()
        frames.append(frame_path)

    # Compile GIF
    with imageio.get_writer(save_path, mode='I', duration=duration / 1000) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    # Clean up frame images
    for f in frames:
        os.remove(f)
    os.rmdir(image_folder)

    return save_path


import plotly.graph_objects as go
import pandas as pd

def visualize_signal(pnl_series: pd.Series, title="Strategy PnL"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pnl_series.index,
        y=pnl_series / 100,  # Convert to decimal for percentage formatting
        name="Cumulative Return",
        line=dict(color="green")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(
            title="Cumulative Return (%)",
            tickformat=".0%",  # Formats ticks as percentages (e.g. 0.85 -> 85%)
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
        ),
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", x=0, y=1.1)
    )

    return fig

