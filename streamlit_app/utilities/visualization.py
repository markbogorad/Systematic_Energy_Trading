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
        x=df["Date"],
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

    Parameters:
    - df: DataFrame with 'date' column and futures contract columns.
    - sheet_name: Name of the commodity (used for the chart title).
    - save_path: Filepath for the final GIF.
    - duration: Frame duration in milliseconds.
    - max_frames: Maximum number of frames in the GIF.
    """
    image_folder = "gif_frames"
    os.makedirs(image_folder, exist_ok=True)

    frames = []
    num_contracts = df.shape[1] - 1  # Exclude 'date' column
    total_rows = len(df)

    step = max(1, total_rows // max_frames)
    selected_rows = df.iloc[::step].reset_index(drop=True)

    all_contracts_patch = mpatches.Patch(color='blue', label='All Contracts')
    winter_patch = mpatches.Patch(color='red', label='Winter Contracts')

    for i, row in selected_rows.iterrows():
        start_month = row["date"]
        contract_months = [start_month + pd.DateOffset(months=j) for j in range(num_contracts)]
        contract_labels = [month.strftime("%Y-%m") for month in contract_months]
        prices = row.iloc[1:].values  # all but date

        plt.figure(figsize=(12, 6))
        plt.plot(contract_labels, prices, marker='o', linestyle='-', color='blue')

        for idx, month in enumerate(contract_months):
            if month.month in [12, 1, 2]:
                plt.scatter(contract_labels[idx], prices[idx], color='red', s=100)

        plt.xlabel("Future Contract Month")
        plt.ylabel("Price")
        plt.title(f"Futures Strip Over Time ({sheet_name})\nDate: {row['date'].strftime('%Y-%m-%d')}")
        plt.xticks(rotation=45)
        plt.ylim(df.iloc[:, 1:].min().min() - 1, df.iloc[:, 1:].max().max() + 1)
        plt.legend(handles=[all_contracts_patch, winter_patch])

        frame_path = os.path.join(image_folder, f"frame_{i:03d}.png")
        plt.savefig(frame_path, bbox_inches="tight")
        plt.close()
        frames.append(frame_path)

    with imageio.get_writer(save_path, mode='I', duration=duration / 1000) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    for f in frames:
        os.remove(f)
    os.rmdir(image_folder)

    return save_path

def visualize_signal(df, price_col, signal_series, title="Strategy Signal"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[price_col],
        name="Price", line=dict(color='blue')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_col,
        height=400,
        template="plotly_white",
        showlegend=True
    )

    return fig
