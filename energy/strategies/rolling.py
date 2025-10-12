# energy/strategies/rolling.py
import numpy as np
import pandas as pd
from energy.analytics.metrics import metrics

import numpy as np
import pandas as pd

def rolling_pnl(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    front_col: str = "F1",
    next_col: str = "F2",
    roll_window: int = 6,
) -> pd.DataFrame:

    df = prices.copy()
    idx = df.index

    # --- Guard rails ---
    expiry_calendar = pd.DatetimeIndex(expiry_calendar).sort_values().unique()
    if len(expiry_calendar) == 0:
        raise ValueError("Expiry calendar is empty.")
    last_date = idx.values[-1]

    # --- Map each date to the next expiry (>= date) ---
    next_exp_idx = np.searchsorted(expiry_calendar.values, idx.values, side="left")
    next_exp_idx = np.clip(next_exp_idx, 0, len(expiry_calendar) - 1)

    # If date is strictly after the matched expiry, bump to the following one (unless at end)
    after_mask = idx.values > expiry_calendar.values[next_exp_idx]
    next_exp_idx = next_exp_idx + after_mask.astype(int)
    final_idx = len(expiry_calendar) - 1
    next_exp_idx[next_exp_idx > final_idx] = final_idx  # cap

    # --- Identify the actual next-expiry date for each row ---
    next_exp_dates = expiry_calendar.values[next_exp_idx]

    # --- Position of that next expiry in the trading index ---
    exp_pos = np.searchsorted(idx.values, next_exp_dates, side="left")

    # --- dte: expiry day = 1 ---
    dte = (exp_pos - np.arange(len(idx))) + 1
    dte = dte.astype(float)

    has_in_sample_next = next_exp_dates <= last_date
    dte[~has_in_sample_next] = np.inf

    df["days_to_expiry"] = dte

    # --- Price arrays ---
    f1 = df[front_col].to_numpy()
    f2 = df[next_col].to_numpy()

    pnl = np.zeros(len(df))
    held = np.empty(len(df), dtype=object)

    # --- Flags (Excel aligned) ---
    roll_window_flag = (dte <= roll_window) & (dte > 1) & np.isfinite(dte)
    roll_day_flag = ((dte == roll_window) & np.isfinite(dte)).astype(int)
    post_expiry_flag = np.zeros(len(df), dtype=int)

    # --- Main loop ---
    held[0] = "F1"

    for t in range(1, len(df)):
        dte_y = dte[t - 1]

        if np.isfinite(dte_y):
            # post expiry = first day after expiry
            if dte_y == 1:
                post_expiry_flag[t] = 1

        # --- Pure price PnL (no costs) ---
        if np.isfinite(dte_y) and dte_y == 1:
            # day after expiry: jump from old F2 to new F1
            pnl[t] = f1[t] - f2[t - 1]
            held[t] = "F1"
        elif np.isfinite(dte_y) and dte_y <= roll_window:
            # inside roll window, hold F2
            pnl[t] = f2[t] - f2[t - 1]
            held[t] = "F2"
        else:
            # normal days
            pnl[t] = f1[t] - f1[t - 1]
            held[t] = "F1"

    # --- Finalize ---
    df["daily_pnl"] = pnl
    df["held_contract"] = held
    df["roll_window_flag"] = roll_window_flag.astype(int)
    df["roll_day_flag"] = roll_day_flag
    df["post_expiry_flag"] = post_expiry_flag

    # Interface consistency
    df["t_cost"] = 0.0
    df["net_pnl"] = df["daily_pnl"]

    return df






def roll_EL(
    rolled_df: pd.DataFrame,
    prices: pd.DataFrame,
    front_col: str = "F1",
    t_cost: float = 0.01,
) -> pd.DataFrame:

    if "daily_pnl" not in rolled_df.columns or "roll_day_flag" not in rolled_df.columns:
        raise ValueError("rolled_df must come from rolling_pnl() (needs daily_pnl & roll_day_flag).")

    df = rolled_df.copy()
    idx = df.index
    n = len(df)
    df["t_cost"] = 0.0

    # --- Entry and exit costs (1× each) ---
    df.loc[idx[0], "t_cost"] -= abs(t_cost)
    df.loc[idx[-1], "t_cost"] -= abs(t_cost)

    # --- Per-roll costs (2× each roll, applied pre-roll) ---
    roll_idxs = np.flatnonzero(df["roll_day_flag"].values == 1)
    for r in roll_idxs:
        target = max(0, r - 1)  # pre-roll day
        df.iat[target, df.columns.get_loc("t_cost")] -= 2.0 * abs(t_cost)

    # --- Build equity line ---
    eq = np.zeros(n)
    eq[0] = prices.at[idx[0], front_col] + df.iat[0, df.columns.get_loc("t_cost")]

    for i in range(1, n):
        eq[i] = eq[i - 1] + df.iat[i, df.columns.get_loc("daily_pnl")] + df.iat[i, df.columns.get_loc("t_cost")]

    df["equity_line"] = np.round(eq, 8)
    df["net_pnl"] = df["daily_pnl"] + df["t_cost"]

    return df






class RollingStrategy:
    def __init__(self, prices, expiry_calendar, front_col="F1", next_col="F2"):
        self.prices = prices
        self.expiry_calendar = expiry_calendar
        self.front_col = front_col
        self.next_col = next_col
        self._rolled = None
        self._equity = None

    def pnl(self, roll_window=6):
        """Compute pure price-based rolling PnL (no costs)."""
        self._rolled = rolling_pnl(
            self.prices,
            self.expiry_calendar,
            front_col=self.front_col,
            next_col=self.next_col,
            roll_window=roll_window,
        )
        return self._rolled

    def equity(
        self,
        roll_window=6,
        t_cost=0.01,
    ):
        """Apply costs and construct equity line."""
        rolled = self._rolled if self._rolled is not None else self.pnl(roll_window)
        self._equity = roll_EL(
            rolled,
            self.prices,
            front_col=self.front_col,
            t_cost=t_cost,
        )
        return self._equity

    def metrics(self, contracts=1, units=1000):
        """Compute summary statistics using global metrics()."""
        if self._equity is None:
            raise ValueError("Must call .equity() before .metrics().")
        return metrics(self._equity, contracts=contracts, units=units)