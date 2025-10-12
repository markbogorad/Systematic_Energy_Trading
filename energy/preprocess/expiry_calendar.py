from pathlib import Path
import pandas as pd

def expiry_calendar(ticker: str, calendar_path: str | Path = "../data/expiry_calendars.xlsx") -> pd.DatetimeIndex:

    df = pd.read_excel(calendar_path, header=0)
    ticker = ticker.upper().strip()

    if ticker not in df.columns:
        raise ValueError(f"Ticker '{ticker}' not found in expiry calendar file.")

    # Drop NaNs and convert to datetime
    expiries = pd.to_datetime(df[ticker].dropna().astype(str), errors='coerce')
    expiries = expiries.dropna().sort_values().unique()

    return pd.DatetimeIndex(expiries)