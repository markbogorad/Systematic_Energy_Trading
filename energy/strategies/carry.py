import pandas as pd
import numpy as np

def carry(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    end_col: str = "F4",
    t_cost: float = 0.01,
    epsilon: float = 0.0,
) -> pd.DataFrame:
    df = prices.copy()
    diff = df[front_col] - df[end_col]
    df["signal"] = np.where(diff > epsilon, 1, np.where(diff < -epsilon, -1, 0))

    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()
    pnl_df["signal"] = df["signal"].reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)
    pnl_df["carry_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    delta = (pnl_df["signal"] - pnl_df["signal"].shift(1)).abs().fillna(0)
    pnl_df["sig_cost_mult"] = np.select([delta == 0, delta == 1, delta == 2], [0, 1, 2], default=0)
    roll_mult = pnl_df["roll_day_flag"] * 2

    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)
    flat_overlap = (pnl_df["roll_day_flag"] == 1) & (pnl_df["signal_lag"] != 0) & (pnl_df["signal"] == 0)
    combined[flat_overlap] = 1
    pnl_df["total_cost_mult"] = combined

    if len(pnl_df) > 0:
        pnl_df.iloc[0, pnl_df.columns.get_loc("total_cost_mult")] = max(pnl_df.iloc[0]["total_cost_mult"], 1)
        pnl_df.iloc[-1, pnl_df.columns.get_loc("total_cost_mult")] = max(pnl_df.iloc[-1]["total_cost_mult"], 1)

    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)
    pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * t_cost
    pnl_df["t_cost"] = pnl_df["sig_t_cost"] + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    pnl_df["net_pnl"] = pnl_df["carry_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    pnl_df = pnl_df.rename(columns={"carry_raw": "daily_pnl", "roll_day_flag": "roll_flag"})[
        ["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line"]
    ]

    return pnl_df