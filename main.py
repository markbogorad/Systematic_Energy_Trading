import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scripts.data_loader import DataLoader
from scripts.visualization import generate_futures_gif
import numpy as np
import imageio as im

# Streamlit App Title
st.title("NGL Statistical Arbitrage Dashboard")

# File Path for Data
file_path = "Data/NGLs.xlsx"
loader = DataLoader(file_path)

# Sidebar Selection
st.sidebar.header("Data Selection")
xls = pd.ExcelFile(file_path)
all_sheets = xls.sheet_names
selected_sheet = st.sidebar.selectbox("Select a Sheet", all_sheets)

# Load Data
df = loader.load_data(selected_sheet)

if not df.empty:
    st.write(f"### Data from {selected_sheet}")
    st.dataframe(df.head(20))

    # Select a date for manual plotting
    selected_date = st.sidebar.selectbox("Select a Date", df.index.strftime("%Y-%m-%d"))

    # Convert back to datetime index
    selected_date = pd.to_datetime(selected_date)

    # Plot selected date's futures strip
    if selected_date in df.index:
        plt.figure(figsize=(8, 5))
        future_values = df.loc[selected_date].values
        plt.plot(np.arange(len(future_values)), future_values, marker='o', linestyle='-')

        plt.xlabel("Future Contract (Index)")
        plt.ylabel("Price")
        plt.title(f"Futures Strip on {selected_date.date()}")
        st.pyplot(plt)

    # Generate GIF and show it
    gif_path = "futures_strip_animation.gif"
    generate_futures_gif(df, output_path=gif_path, frame_rate=0.5)

    st.image(gif_path, caption="Futures Strip Animation", use_column_width=True)

else:
    st.warning(f"No data available for {selected_sheet}.")
