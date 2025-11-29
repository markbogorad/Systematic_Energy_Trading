import pandas as pd
import numpy as np

def carry(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    end_col: str = "F4",
    t_cost: float = 0.01,        # absolute cost in ORIGINAL quote units
    pct_t_cost: float | None = None,  # fraction of |daily_pnl|, e.g. 0.001 = 10 bps
    epsilon: float = 0.0,
) -> pd.DataFrame:
    df = prices.copy()
    diff = df[front_col] - df[end_col]
    df["signal"] = np.where(diff > epsilon, 1, np.where(diff < -epsilon, -1, 0))

    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()
    pnl_df["signal"]     = df["signal"].reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # raw carry PnL: position_{t-1} * underlying daily_pnl
    pnl_df["carry_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # signal-change cost multiplier: 0,1,2 turns
    delta = (pnl_df["signal"] - pnl_df["signal_lag"]).abs().fillna(0)
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta == 2],
        [0, 1, 2],
        default=0,
    )

    # roll multiplier from underlying roll engine
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

    # flat overlap: exiting to flat on roll day
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined[flat_overlap] = 1

    pnl_df["total_cost_mult"] = combined

    # enforce at least one turn on first/last obs
    if len(pnl_df) > 0:
        i0, i1 = pnl_df.index[0], pnl_df.index[-1]
        pnl_df.loc[i0, "total_cost_mult"] = max(pnl_df.loc[i0, "total_cost_mult"], 1)
        pnl_df.loc[i1, "total_cost_mult"] = max(pnl_df.loc[i1, "total_cost_mult"], 1)

    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)

    # --- ABSOLUTE vs PERCENT T-COST, with normalization ---
    norm_scale = prices.attrs.get("norm_scale", 1.0)

    if pct_t_cost is not None and pct_t_cost > 0:
        # percentage of underlying roll daily_pnl (per unit)
        base_cost = pct_t_cost * pnl_df["daily_pnl"].abs()
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * base_cost
    else:
        # absolute cost per unit, scaled into normalized units
        abs_tc = t_cost * norm_scale
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * abs_tc

    # combine strategy t_cost with underlying roll cost
    pnl_df["t_cost"] = (
        pnl_df["sig_t_cost"]
        + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    )

    # net PnL & equity
    pnl_df["net_pnl"] = pnl_df["carry_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    # trade count = total number of turns that day (0,1,2)
    pnl_df["trade_count"] = pnl_df["total_cost_mult"]

    # standard output schema
    pnl_df = pnl_df.rename(
        columns={"carry_raw": "daily_pnl", "roll_day_flag": "roll_flag"}
    )[["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line", "trade_count"]]

    return pnl_df
