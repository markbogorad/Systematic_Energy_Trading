import numpy as np
import pandas as pd

def metrics(df: pd.DataFrame, contracts=1, units=1000) -> pd.Series:

    if "net_pnl" not in df.columns or "t_cost" not in df.columns:
        raise ValueError("DataFrame must include 'net_pnl' and 't_cost'.")

    idx = df.index
    years = (idx[-1] - idx[0]).days / 365
    obs_per_year = len(idx) / years if years > 0 else np.nan
    sqrt_af = np.sqrt(obs_per_year) if obs_per_year > 0 else np.nan

    total_pnl = df["net_pnl"].sum() * contracts * units
    total_cost = df["t_cost"].sum() * contracts * units
    apl_unit = (total_pnl / (contracts * units)) / years if years > 0 else np.nan

    daily_std = df["net_pnl"].std()
    ann_std = daily_std * np.sqrt(250)
    sharpe = apl_unit / ann_std if ann_std and ann_std != 0 else np.nan

    pnl_index = 100 + df["net_pnl"].cumsum()
    roll_max = pnl_index.cummax()
    drawdown = (pnl_index / roll_max - 1.0).min()
    rod = -apl_unit / drawdown if drawdown and drawdown != 0 else np.nan

    out = pd.Series({
        "Total PnL": round(total_pnl, 4),
        "Total Cost": round(total_cost, 4),
        "APL/unit": round(apl_unit, 4),
        "Std Dev (ann.)": round(ann_std, 4),
        "Sharpe": round(sharpe, 4),
        "Drawdown": round(drawdown, 4),
        "RoD": round(rod, 4),
        "Years": round(years, 4),
    })

    # Apply local display setting
    pd.options.display.float_format = "{:,.4f}".format

    return out
