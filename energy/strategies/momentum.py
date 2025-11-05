import pandas as pd
import numpy as np
from collections.abc import Iterable

# ---------------------------------------------------------------------
# ORIGINAL — kept exactly as you wrote it
# ---------------------------------------------------------------------
def momentum(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    short_ma: int = 1,
    long_ma: int = 20,
    t_cost: float = 0.01,
    epsilon: float = 0.0,
) -> pd.DataFrame:
    df = prices.copy()
    short_avg = df[front_col].rolling(window=short_ma, min_periods=short_ma).mean()
    long_avg = df[front_col].rolling(window=long_ma, min_periods=long_ma).mean()
    diff = short_avg - long_avg
    df["signal"] = np.where(diff > epsilon, 1, np.where(diff < -epsilon, -1, 0))

    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()
    pnl_df["signal"] = df["signal"].reindex(pnl_df.index).ffill()
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    start_date = df.index[long_ma - 1] if len(df) >= long_ma else df.index[0]
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    pnl_df["mom_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    delta = (pnl_df["signal"] - pnl_df["signal"].shift(1)).abs().fillna(0)
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta == 2], [0, 1, 2], default=0
    )
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined[flat_overlap] = 1
    pnl_df["total_cost_mult"] = combined

    if not pnl_df.empty:
        pnl_df.iloc[0, pnl_df.columns.get_loc("total_cost_mult")] = max(
            pnl_df.iloc[0]["total_cost_mult"], 1
        )
        pnl_df.iloc[-1, pnl_df.columns.get_loc("total_cost_mult")] = max(
            pnl_df.iloc[-1]["total_cost_mult"], 1
        )

    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)
    pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * t_cost
    pnl_df["t_cost"] = pnl_df["sig_t_cost"] + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0

    pnl_df["net_pnl"] = pnl_df["mom_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    pnl_df = pnl_df.rename(columns={"mom_raw": "daily_pnl", "roll_day_flag": "roll_flag"})[
        ["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line"]
    ]
    return pnl_df


# ---------------------------------------------------------------------
# Helper: apply a PRE-COMPUTED binary/ternary signal to your exact PnL/TCost engine
# (identical mechanics to the original function)
# ---------------------------------------------------------------------
def _apply_signal_to_rolled(
    signal: pd.Series,
    rolled_df: pd.DataFrame,
    start_date,              # first date at which the signal is “live” (e.g., max lookback - 1)
    t_cost: float,
) -> pd.DataFrame:
    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()

    # align and build lagged signal (trade on next bar)
    pnl_df["signal"] = signal.reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # flat before warm-up; no roll costs before start_date either
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    # raw strategy pnl
    pnl_df["mom_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # signal-change vs roll multipliers
    delta = (pnl_df["signal"] - pnl_df["signal"].shift(1)).abs().fillna(0)
    pnl_df["sig_cost_mult"] = np.select([delta == 0, delta == 1, delta == 2], [0, 1, 2], default=0)
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

    # flat overlap edge-case
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined[flat_overlap] = 1
    pnl_df["total_cost_mult"] = combined

    # first/last bar: enforce at least one turn
    if not pnl_df.empty:
        i0, i1 = pnl_df.index[0], pnl_df.index[-1]
        pnl_df.loc[i0, "total_cost_mult"] = max(pnl_df.loc[i0, "total_cost_mult"], 1)
        pnl_df.loc[i1, "total_cost_mult"] = max(pnl_df.loc[i1, "total_cost_mult"], 1)

    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)
    pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * t_cost
    pnl_df["t_cost"] = pnl_df["sig_t_cost"] + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0

    pnl_df["net_pnl"] = pnl_df["mom_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    return pnl_df.rename(columns={"mom_raw": "daily_pnl", "roll_day_flag": "roll_flag"})[
        ["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line"]
    ]


def momentum_ma_simple(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    lookbacks=(20, 60, 120, 180, 250),
    t_cost: float = 0.01,
    epsilon: float = 0.0,
    warm_start: bool = True,   # start trading immediately with partial windows
) -> pd.DataFrame:
    px = prices[front_col].astype(float)

    sig_parts = []
    for L in lookbacks:
        # warm-start with partial windows
        ma = px.rolling(window=L, min_periods=1 if warm_start else L).mean()
        sig_parts.append((px / ma - 1.0).rename(f"s_{L}"))

    sig_df = pd.concat(sig_parts, axis=1)

    if warm_start:
        # mean over available horizons each day
        sig_agg = sig_df.mean(axis=1, skipna=True)
        start_date = prices.index[0]
    else:
        sig_agg = sig_df.dropna().mean(axis=1)
        max_L = max(lookbacks)
        start_date = prices.index[max_L - 1] if len(prices.index) >= max_L else prices.index[0]

    signal = pd.Series(
        np.where(sig_agg > epsilon, 1, np.where(sig_agg < -epsilon, -1, 0)),
        index=sig_agg.index,
        name="signal",
    )
    return _apply_signal_to_rolled(signal, rolled_df, start_date, t_cost)


def momentum_ma_ewma(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    lookbacks=(20, 60, 120, 180, 250),
    lam: float = 0.05,
    t_cost: float = 0.01,
    epsilon: float = 0.0,
    warm_start: bool = True,   # start trading immediately with partial windows
) -> pd.DataFrame:
    px = prices[front_col].astype(float)

    sig_parts = []
    for L in lookbacks:
        ma = px.rolling(window=L, min_periods=1 if warm_start else L).mean()
        sig_parts.append((px / ma - 1.0).rename(f"s_{L}"))

    sig_df = pd.concat(sig_parts, axis=1)

    # base EW weights by order (shorter horizons first)
    n = len(lookbacks)
    w = np.exp(-lam * np.arange(n, dtype=float))

    if warm_start:
        # reweight over available horizons each day
        mask = sig_df.notna().astype(float)
        w_row = (mask * w).div((mask * w).sum(axis=1), axis=0).fillna(0.0)
        sig_agg = (sig_df.fillna(0.0) * w_row).sum(axis=1)
        start_date = prices.index[0]
    else:
        w = w / w.sum()
        sig_agg = (sig_df * w).dropna().sum(axis=1)
        max_L = max(lookbacks)
        start_date = prices.index[max_L - 1] if len(prices.index) >= max_L else prices.index[0]

    signal = pd.Series(
        np.where(sig_agg > epsilon, 1, np.where(sig_agg < -epsilon, -1, 0)),
        index=sig_agg.index,
        name="signal",
    )
    return _apply_signal_to_rolled(signal, rolled_df, start_date, t_cost)
