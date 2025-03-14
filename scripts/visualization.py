import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def generate_futures_gif(df, output_path="futures_strip_animation.gif", frame_rate=0.5):
    """Generates an animated GIF showing futures curves over time."""
    os.makedirs("temp_frames", exist_ok=True)
    frames = []

    plot_dates = df.index[::30]  # Sample every 30 days to reduce frame count

    for i, date in enumerate(plot_dates):
        plt.figure(figsize=(8, 5))
        future_values = df.loc[date].values
        plt.plot(np.arange(len(future_values)), future_values, marker='o', linestyle='-')

        plt.xlabel("Future Contract (Index)")
        plt.ylabel("Price")
        plt.title(f"Futures Strip on {date.date()}")

        frame_path = f"temp_frames/frame_{i}.png"
        plt.savefig(frame_path)
        plt.close()
        frames.append(imageio.imread(frame_path))

    # Save as GIF
    imageio.mimsave(output_path, frames, duration=frame_rate)
    print(f"GIF saved as {output_path}")
