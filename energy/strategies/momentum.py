import pandas as pd
import numpy as np
from collections.abc import Iterable

def price_momentum(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    short_ma: int = 1,
    long_ma: int = 20,
    t_cost: float = 0.01,       # absolute cost in ORIGINAL quote units
    pct_t_cost=None,            # if set, fraction of |daily_pnl| (e.g. 0.001 = 10 bps)
    epsilon: float = 0.0,
    ma_pairs=None,              # list of (short, long) pairs
    weights=None,               # optional weights for ma_pairs
) -> pd.DataFrame:
    """
    Price-based momentum on the underlying futures tenor (front_col),
    with PnL taken from rolled_df (rolled engine), and t-costs applied
    on top of the roll t_cost.

    - If pct_t_cost is None/0: use absolute t_cost, scaled by prices.attrs["norm_scale"].
    - If pct_t_cost > 0: ignore t_cost and charge pct_t_cost * |daily_pnl|
      per 'turn' (0,1,2) on top of roll costs.
    """
    df = prices.copy()
    px = df[front_col].astype(float)

    # ----------------------------
    # 1) BUILD RAW SCORE + start_date
    # ----------------------------
    if ma_pairs is None:
        # Single MA crossover
        short_avg = px.rolling(window=short_ma, min_periods=short_ma).mean()
        long_avg  = px.rolling(window=long_ma,  min_periods=long_ma).mean()
        raw_score = short_avg - long_avg   # preserve old behavior

        start_date = (
            df.index[long_ma - 1] if len(df) >= long_ma else df.index[0]
        )
    else:
        # Multi-crossover logic
        ma_pairs = list(ma_pairs)

        if weights is None:
            weights = np.ones(len(ma_pairs), dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(ma_pairs):
                raise ValueError("weights must have same length as ma_pairs")

        weights = weights / weights.sum()

        score_cols = []
        max_L = 0

        for (s, l) in ma_pairs:
            if s <= 0 or l <= 0 or s >= l:
                raise ValueError(
                    f"Invalid MA pair (short={s}, long={l}); require 0 < short < long"
                )

            max_L = max(max_L, l)

            ma_s = px.rolling(window=s, min_periods=s).mean()
            ma_l = px.rolling(window=l, min_periods=l).mean()

            # continuous crossover signal
            score = (ma_s / ma_l - 1.0)
            score_cols.append(score)

        score_df = pd.concat(score_cols, axis=1)

        # weighted average of all crossover scores
        raw_score = (score_df * weights).sum(axis=1)

        # first date when ALL long MAs are fully formed
        start_date = (
            df.index[max_L - 1] if len(df) >= max_L else df.index[0]
        )

    # ----------------------------
    # 2) DISCRETIZE TO {-1,0,+1}
    # ----------------------------
    df["signal"] = 0
    df.loc[raw_score.index, "signal"] = np.where(
        raw_score > epsilon,
        1,
        np.where(raw_score < -epsilon, -1, 0),
    )

    # ----------------------------
    # 3) PnL / T-COST ENGINE 
    # ----------------------------
    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()

    # align and trade on next bar
    pnl_df["signal"]     = df["signal"].reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # flat & no roll costs before warm-up
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    # raw strategy pnl: position_{t-1} * underlying daily_pnl
    pnl_df["mom_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # signal-change cost multiplier
    delta = (pnl_df["signal"] - pnl_df["signal_lag"]).abs().fillna(0)
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta == 2],
        [0, 1, 2],
        default=0,
    )

    # roll multiplier (existing logic)
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
        pnl_df.loc[i0, "total_cost_mult"] = max(
            pnl_df.loc[i0, "total_cost_mult"], 1
        )
        pnl_df.loc[i1, "total_cost_mult"] = max(
            pnl_df.loc[i1, "total_cost_mult"], 1
        )

    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)

    # ----------------------------
    # 3a) ABSOLUTE vs PERCENT T-COST (with normalization)
    # ----------------------------
    norm_scale = prices.attrs.get("norm_scale", 1.0)

    if pct_t_cost is not None and pct_t_cost > 0:
        # percentage of underlying roll daily_pnl (per unit)
        base_cost = pct_t_cost * pnl_df["daily_pnl"].abs()
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * base_cost
    else:
        # absolute cost in original units → scale into normalized space
        abs_tc = t_cost * norm_scale
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * abs_tc

    # combine strategy t_cost with underlying roll cost
    pnl_df["t_cost"] = (
        pnl_df["sig_t_cost"]
        + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    )
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0

    # trade count = total number of turns that day (0, 1, or 2)
    pnl_df["trade_count"] = pnl_df["total_cost_mult"]

    # net PnL & equity
    pnl_df["net_pnl"] = pnl_df["mom_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    return pnl_df.rename(
        columns={"mom_raw": "daily_pnl", "roll_day_flag": "roll_flag"}
    )[["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line", "trade_count"]]


def momentum(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    short_ma: int = 1,
    long_ma: int = 20,
    t_cost: float = 0.01,
    pct_t_cost=None,
    epsilon: float = 0.0,
    ma_pairs=None,
    weights=None,
) -> pd.DataFrame:
    """
    Roll-adjusted price momentum.

    This is now just a thin wrapper around price_momentum so that:
    - we use the price MA signal on the appropriate tenor (front_col),
    - PnL comes from rolled_df (roll-adjusted engine),
    - t-costs use the new normalization + pct_t_cost logic.
    """
    return price_momentum(
        prices=prices,
        rolled_df=rolled_df,
        front_col=front_col,
        short_ma=short_ma,
        long_ma=long_ma,
        t_cost=t_cost,
        pct_t_cost=pct_t_cost,
        epsilon=epsilon,
        ma_pairs=ma_pairs,
        weights=weights,
    )
