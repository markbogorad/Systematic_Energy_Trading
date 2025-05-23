import pandas as pd
import numpy as np

def compute_strategy_metrics(pnl_series: pd.Series, freq: str = 'D') -> dict:
    """
    Compute key strategy performance metrics from a cumulative PnL time series.
    
    Parameters:
    - pnl_series (pd.Series): Indexed by datetime, cumulative PnL.
    - freq (str): Frequency of data. 'D' for daily. Affects annualization.
    
    Returns:
    - dict of metrics
    """
    # Clean and ensure sorted datetime index
    pnl_series = pnl_series.dropna().sort_index()

    if pnl_series.empty or pnl_series.isna().all():
        return {
            "Total P&L": np.nan,
            "Annualized P&L": np.nan,
            "Sharpe Ratio": np.nan,
            "Annualized Std Dev": np.nan,
            "High Water Mark": np.nan,
            "Max Drawdown": np.nan,
            "Return on Drawdown": np.nan
        }

    daily_returns = pnl_series.diff().dropna()
    ann_factor = 252 if freq == 'D' else 12 if freq == 'M' else 1

    # Total and annualized PnL
    total_pnl = pnl_series.iloc[-1] - pnl_series.iloc[0]

    num_years = (pnl_series.index[-1] - pnl_series.index[0]).days / 365.25
    annual_pnl = total_pnl / num_years if num_years > 0 else np.nan

    # Annualized volatility and Sharpe
    ann_std = daily_returns.std() * np.sqrt(ann_factor)
    sharpe = annual_pnl / ann_std if ann_std != 0 else np.nan

    # Drawdown calculations
    hwm = pnl_series.cummax()
    drawdown = hwm - pnl_series
    max_dd = drawdown.max()
    hwm_val = hwm.max()
    return_on_dd = annual_pnl / max_dd if max_dd != 0 else np.nan

    return {
        "Total P&L": round(total_pnl, 2),
        "Annualized P&L": round(annual_pnl, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Annualized Std Dev": round(ann_std, 2),
        "High Water Mark": round(hwm_val, 2),
        "Max Drawdown": round(max_dd, 2),
        "Return on Drawdown": round(return_on_dd, 2)
    }
