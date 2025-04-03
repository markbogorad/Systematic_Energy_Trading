import plotly.graph_objects as go
import matplotlib as plt
import numpy as np
import imageio
import os

def generate_futures_gif(df, output_path="futures_strip_animation.gif", frame_rate=0.5, sample_rate=30):
    """
    Generates an animated GIF showing futures curves over time.

    Parameters:
    - df: DataFrame with dates as index and futures contracts as columns
    - output_path: Where to save the final GIF
    - frame_rate: Duration per frame (in seconds)
    - sample_rate: How often to sample dates (e.g., every 30 days)
    """
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frames = []

    plot_dates = df.index[::sample_rate]

    for i, date in enumerate(plot_dates):
        fig, ax = plt.subplots(figsize=(8, 5))
        y = df.loc[date].values
        x = np.arange(len(y))

        ax.plot(x, y, marker='o', linestyle='-')
        ax.set_xlabel("Future Contract (Index)")
        ax.set_ylabel("Price")
        ax.set_title(f"Futures Strip on {date.strftime('%Y-%m-%d')}")
        ax.grid(True)

        frame_path = f"{temp_dir}/frame_{i}.png"
        plt.savefig(frame_path)
        plt.close()
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(output_path, frames, duration=frame_rate)

    # Optional: clean up temporary frames
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    print(f"GIF saved at {output_path}")


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
