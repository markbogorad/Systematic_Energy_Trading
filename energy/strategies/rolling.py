# energy/strategies/rolling.py
import numpy as np
import pandas as pd
from typing import List
from energy.analytics.metrics import metrics

# =============================================================================
# Helpers
# =============================================================================
def _require_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _next_expiry_for_each_date(
    idx: pd.DatetimeIndex, expiry_calendar: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    exp = pd.DatetimeIndex(expiry_calendar).sort_values().unique()
    if len(exp) == 0:
        raise ValueError("Expiry calendar is empty.")

    next_exp_idx = np.searchsorted(exp.values, idx.values, side="left")
    next_exp_idx = np.clip(next_exp_idx, 0, len(exp) - 1)

    # if we're already after expiry on that date -> move to next
    after = idx.values > exp.values[next_exp_idx]
    next_exp_idx = np.clip(next_exp_idx + after.astype(int), 0, len(exp) - 1)

    return pd.DatetimeIndex(exp.values[next_exp_idx])


def _first_trading_day_of_month_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    n = len(idx)
    out = np.zeros(n, dtype=int)
    if n > 1:
        m = idx.month.to_numpy()
        out[1:] = (m[1:] != m[:-1]).astype(int)
    return out


def _post_expiry_flags(idx: pd.DatetimeIndex, next_exp_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    1 on the first trading day strictly AFTER the relevant expiry.
    This is the *source of truth* flag — everything else should look at t-1.
    """
    n = len(idx)
    out = np.zeros(n, dtype=int)
    for t in range(1, n):
        if idx[t - 1] < next_exp_dates[t - 1] <= idx[t]:
            out[t] = 1
    return out


def _expiry_today_flags(idx: pd.DatetimeIndex, next_exp_dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(idx)
    out = np.zeros(n, dtype=int)
    for t in range(1, n):
        if next_exp_dates[t - 1] == idx[t]:
            out[t] = 1
    return out


# =============================================================================
# 1) Synchronous T-N / window roll (this one was already correct)
# =============================================================================
def rolling_pnl(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    front_col: str = "F1",
    next_col: str = "F2",
    roll_window: int = 5,
) -> pd.DataFrame:

    df = prices.copy()
    idx = df.index

    exp = pd.DatetimeIndex(expiry_calendar).sort_values().unique()
    if len(exp) == 0:
        raise ValueError("Expiry calendar is empty.")

    next_exp_idx = np.searchsorted(exp.values, idx.values, side="left")
    next_exp_idx = np.clip(next_exp_idx, 0, len(exp) - 1)
    next_exp_dates = exp.values[next_exp_idx]
    exp_pos = np.searchsorted(idx.values, next_exp_dates, side="left")

    arange_idx = np.arange(len(idx))
    dte = (exp_pos - arange_idx - 1).astype(float)
    last_date = idx.values[-1]
    valid = next_exp_dates <= last_date
    dte[~valid] = np.inf

    _require_cols(df, [front_col, next_col])
    f1 = df[front_col].to_numpy(float)
    f2 = df[next_col].to_numpy(float)

    n = len(df)
    daily_pnl = np.zeros(n, dtype=float)
    held = np.empty(n, dtype=object)
    roll_day_flag = np.zeros(n, dtype=int)
    post_expiry_flag = np.zeros(n, dtype=int)

    daily_pnl[0] = 0.0
    held[0] = front_col

    for t in range(1, n):
        d = dte[t - 1]

        if np.isfinite(d) and d == -1:
            daily_pnl[t] = f1[t] - f2[t - 1]
            held[t] = front_col
            post_expiry_flag[t] = 1

        elif np.isfinite(d) and d == roll_window:
            daily_pnl[t] = f1[t] - f1[t - 1]
            held[t] = next_col
            roll_day_flag[t] = 1

        elif np.isfinite(d) and 0 <= d < roll_window:
            daily_pnl[t] = f2[t] - f2[t - 1]
            held[t] = next_col

        elif np.isfinite(d) and d > roll_window:
            daily_pnl[t] = f1[t] - f1[t - 1]
            held[t] = front_col

        else:
            daily_pnl[t] = f1[t] - f1[t - 1]
            held[t] = front_col

    df["daily_pnl"] = daily_pnl
    df["held_contract"] = held
    df["roll_day_flag"] = roll_day_flag
    df["post_expiry_flag"] = post_expiry_flag
    df["t_cost"] = 0.0
    df["net_pnl"] = df["daily_pnl"]
    return df


# =============================================================================
# 2) EOM / midmonth family (all now use post_expiry_flag[t-1])
# =============================================================================
def roll_EOM_midmonth_expiry(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    front_col: str = "F1",  # the month that becomes front after expiry
    next_col: str = "F2",   # the month we normally sit in (M2)
) -> pd.DataFrame:
    """
    WTI-style midmonth/EOM, expiry-driven, with t-1 lookback.

    Pattern we want:
        ... M2 ... M2 ... (expiry hits) ... M1 ...  [next bar]  sell M1, buy new M2 ...
    i.e. we LIVE in M2, briefly sit in M1 right after it falls in, then roll back to M2.
    """
    out = prices.copy()
    idx = out.index
    n = len(idx)

    _require_cols(out, [front_col, next_col])

    next_exp_dates = _next_expiry_for_each_date(idx, expiry_calendar)
    post_expiry_flag = _post_expiry_flags(idx, next_exp_dates)

    month = idx.month.to_numpy()
    next_exp_month = next_exp_dates.month.to_numpy()

    f1 = out[front_col].to_numpy(float)  # M1
    f2 = out[next_col].to_numpy(float)   # M2

    daily_pnl = np.zeros(n, dtype=float)
    held = np.empty(n, dtype=object)
    roll_day_flag = np.zeros(n, dtype=int)

    # start in M2, not M1
    daily_pnl[0] = 0.0
    held[0] = next_col

    for t in range(1, n):
        # if yesterday we detected expiry, TODAY we roll out of the old M1 (yday's M2)
        # and back into the new M2
        if post_expiry_flag[t - 1] == 1:
            # sell today's M1 (== yesterday's M2), buy today's M2
            daily_pnl[t] = f1[t] - f2[t - 1]
            held[t] = next_col          # <- end the day in M2 again
            roll_day_flag[t] = 1
            continue

        # otherwise: choose between "before expiry" and "after expiry"
        cur_month = month[t - 1]
        exp_month = next_exp_month[t - 1]

        if exp_month == cur_month:
            # BEFORE midmonth expiry → stay in M2
            daily_pnl[t] = f2[t] - f2[t - 1]
            held[t] = next_col
        else:
            # AFTER midmonth expiry (the old M2 has fallen into M1) → track M1
            daily_pnl[t] = f1[t] - f1[t - 1]
            held[t] = front_col

    out["daily_pnl"] = daily_pnl
    out["held_contract"] = held
    out["roll_day_flag"] = roll_day_flag
    out["post_expiry_flag"] = post_expiry_flag
    out["t_cost"] = 0.0
    out["net_pnl"] = out["daily_pnl"]
    return out



def roll_EOM_NGL(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    mid_col: str = "F3",  # M3
    far_col: str = "F4",  # M4
) -> pd.DataFrame:
    """
    NGL-style EOM, t-1 view:
      - normal: M3[t] - M3[t-1]
      - if yesterday was post-expiry: M3[t] - M4[t-1]
    """
    out = prices.copy()
    idx = out.index
    n = len(idx)

    _require_cols(out, [mid_col, far_col])

    next_exp_dates = _next_expiry_for_each_date(idx, expiry_calendar)
    post_expiry_flag = _post_expiry_flags(idx, next_exp_dates)

    m3 = out[mid_col].to_numpy(float)
    m4 = out[far_col].to_numpy(float)

    daily_pnl = np.zeros(n, dtype=float)
    held = np.empty(n, dtype=object)
    roll_day_flag = np.zeros(n, dtype=int)

    daily_pnl[0] = 0.0
    held[0] = mid_col

    for t in range(1, n):
        if post_expiry_flag[t - 1] == 1:
            daily_pnl[t] = m3[t] - m4[t - 1]
            held[t] = mid_col
            roll_day_flag[t] = 1
        else:
            daily_pnl[t] = m3[t] - m3[t - 1]
            held[t] = mid_col

    out["daily_pnl"] = daily_pnl
    out["held_contract"] = held
    out["roll_day_flag"] = roll_day_flag
    out["post_expiry_flag"] = post_expiry_flag
    out["t_cost"] = 0.0
    out["net_pnl"] = out["daily_pnl"]
    return out


def roll_EOM_EOM_expiry(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    next_col: str = "F2",   # M2
    third_col: str = "F3",  # M3
) -> pd.DataFrame:
    """
    Always-EOM-expiry style, but still using t-1:
      - normal: M2[t] - M2[t-1]
      - if yesterday was post-expiry: M2[t] - M3[t-1]
    """
    out = prices.copy()
    idx = out.index
    n = len(idx)

    _require_cols(out, [next_col, third_col])

    next_exp_dates = _next_expiry_for_each_date(idx, expiry_calendar)
    post_expiry_flag = _post_expiry_flags(idx, next_exp_dates)

    m2 = out[next_col].to_numpy(float)
    m3 = out[third_col].to_numpy(float)

    daily_pnl = np.zeros(n, dtype=float)
    held = np.empty(n, dtype=object)
    roll_day_flag = np.zeros(n, dtype=int)

    daily_pnl[0] = 0.0
    held[0] = next_col

    for t in range(1, n):
        if post_expiry_flag[t - 1] == 1:
            daily_pnl[t] = m2[t] - m3[t - 1]
            held[t] = next_col
            roll_day_flag[t] = 1
        else:
            daily_pnl[t] = m2[t] - m2[t - 1]
            held[t] = next_col

    out["daily_pnl"] = daily_pnl
    out["held_contract"] = held
    out["roll_day_flag"] = roll_day_flag
    out["post_expiry_flag"] = post_expiry_flag
    out["t_cost"] = 0.0
    out["net_pnl"] = out["daily_pnl"]
    return out


def roll_EOM_dynamic_brent(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    front_col: str = "F1",   # M1
    next_col: str = "F2",    # M2 (the one we normally live in)
    third_col: str = "F3",   # M3 (for true EOM rolls)
) -> pd.DataFrame:
    """
    Brent-style dynamic roll with t-1 lookback.

    Idea:
    - If the relevant expiry rolls into a NEW MONTH (true EOM month) → use EOM flavour:
        normal:     M2[t] - M2[t-1]
        post-expiry: M2[t] - M3[t-1]
    - Otherwise (midmonth month) → use the UPDATED midmonth/WTI-style pattern:
        ... M2 ... M2 ... (expiry) ... M1 ... [next bar] sell M1, buy M2 ...
    """
    out = prices.copy()
    idx = out.index
    n = len(idx)

    _require_cols(out, [front_col, next_col, third_col])

    # per-date next expiry + flags
    next_exp_dates = _next_expiry_for_each_date(idx, expiry_calendar)
    post_expiry_flag = _post_expiry_flags(idx, next_exp_dates)

    # detect if this expiry is EOM-like: (expiry + 1 day) is in a new month
    is_eom_next = np.array(
        [((d + pd.offsets.Day(1)).month != d.month) for d in next_exp_dates]
    )

    month = idx.month.to_numpy()
    next_exp_month = next_exp_dates.month.to_numpy()

    f1 = out[front_col].to_numpy(float)   # M1
    f2 = out[next_col].to_numpy(float)    # M2
    f3 = out[third_col].to_numpy(float)   # M3

    daily_pnl = np.zeros(n, dtype=float)
    held = np.empty(n, dtype=object)
    roll_day_flag = np.zeros(n, dtype=int)

    # start: if upcoming expiry is EOM → we live in M2; else midmonth → also live in M2
    daily_pnl[0] = 0.0
    held[0] = next_col

    for t in range(1, n):
        if is_eom_next[t - 1]:
            # ---------------------------------------------------------
            # EOM flavour: same as roll_EOM_EOM_expiry (M2 ↔ M3)
            # ---------------------------------------------------------
            if post_expiry_flag[t - 1] == 1:
                # day AFTER expiry → roll old M2 into M2 (vs M3[t-1])
                daily_pnl[t] = f2[t] - f3[t - 1]
                held[t] = next_col
                roll_day_flag[t] = 1
            else:
                # normal → stay in M2
                daily_pnl[t] = f2[t] - f2[t - 1]
                held[t] = next_col
        else:
            # ---------------------------------------------------------
            # MIDMONTH flavour: use UPDATED WTI-style logic
            # ... M2 ... M2 ... (expiry) ... M1 ... [next] M1→M2 ...
            # ---------------------------------------------------------
            if post_expiry_flag[t - 1] == 1:
                # this is the *day after* expiry → sell M1 (today) vs M2 (yday)
                daily_pnl[t] = f1[t] - f2[t - 1]
                held[t] = next_col      # go back to living in M2
                roll_day_flag[t] = 1
                continue

            cur_month = month[t - 1]
            exp_month = next_exp_month[t - 1]

            if exp_month == cur_month:
                # BEFORE midmonth expiry → stay in M2
                daily_pnl[t] = f2[t] - f2[t - 1]
                held[t] = next_col
            else:
                # AFTER midmonth expiry → track M1 (the fallen M2)
                daily_pnl[t] = f1[t] - f1[t - 1]
                held[t] = front_col

    out["daily_pnl"] = daily_pnl
    out["held_contract"] = held
    out["roll_day_flag"] = roll_day_flag
    out["post_expiry_flag"] = post_expiry_flag
    out["t_cost"] = 0.0
    out["net_pnl"] = out["daily_pnl"]
    return out

# =============================================================================
# 3) Costs & equity
# =============================================================================
def roll_EL(
    rolled_df: pd.DataFrame,
    prices: pd.DataFrame,
    front_col: str = "F1",
    t_cost: float = 0.01,
) -> pd.DataFrame:
    if "daily_pnl" not in rolled_df.columns or "roll_day_flag" not in rolled_df.columns:
        raise ValueError("rolled_df must include 'daily_pnl' and 'roll_day_flag'.")

    df = rolled_df.copy()
    idx = df.index
    n = len(df)

    if "held_contract" in df.columns and pd.notna(df["held_contract"].iloc[0]):
        seed_contract = str(df["held_contract"].iloc[0])
    else:
        seed_contract = front_col

    if seed_contract not in prices.columns:
        raise ValueError(f"Seed contract '{seed_contract}' not found in prices.")

    df["t_cost"] = 0.0

    # entry / exit
    df.iat[0, df.columns.get_loc("t_cost")] -= abs(t_cost)
    df.iat[-1, df.columns.get_loc("t_cost")] -= abs(t_cost)

    # roll costs
    roll_days = np.flatnonzero(df["roll_day_flag"].to_numpy() == 1)
    for r in roll_days:
        df.iat[r, df.columns.get_loc("t_cost")] -= 2.0 * abs(t_cost)

    eq = np.zeros(n, dtype=float)
    seed_price = prices.at[idx[0], seed_contract]
    eq[0] = seed_price + df.iat[0, df.columns.get_loc("t_cost")]

    for t in range(1, n):
        eq[t] = (
            eq[t - 1]
            + df.iat[t, df.columns.get_loc("daily_pnl")]
            + df.iat[t, df.columns.get_loc("t_cost")]
        )

    df["equity_line"] = np.round(eq, 8)
    df["net_pnl"] = df["daily_pnl"] + df["t_cost"]
    return df


# =============================================================================
# 4) Strategy wrapper
# =============================================================================
class RollingStrategy:
    def __init__(self, prices, expiry_calendar, front_col="F1", next_col="F2"):
        self.prices = prices
        self.expiry_calendar = expiry_calendar
        self.front_col = front_col
        self.next_col = next_col
        self._rolled = None
        self._equity = None

    def pnl(self, roll_window=5):
        self._rolled = rolling_pnl(
            self.prices,
            self.expiry_calendar,
            front_col=self.front_col,
            next_col=self.next_col,
            roll_window=roll_window,
        )
        return self._rolled

    def pnl_eom_midmonth(self):
        self._rolled = roll_EOM_midmonth_expiry(
            self.prices,
            self.expiry_calendar,
            front_col=self.front_col,
            next_col=self.next_col,
        )
        return self._rolled

    def pnl_eom_ngl(self, mid_col: str = "F3", far_col: str = "F4"):
        self._rolled = roll_EOM_NGL(
            self.prices,
            self.expiry_calendar,
            mid_col=mid_col,
            far_col=far_col,
        )
        return self._rolled

    def pnl_eom_eom(self, next_col: str = None, third_col: str = "F3"):
        use_next = next_col if next_col is not None else self.next_col
        self._rolled = roll_EOM_EOM_expiry(
            self.prices,
            self.expiry_calendar,
            next_col=use_next,
            third_col=third_col,
        )
        return self._rolled

    def pnl_eom_dynamic(self, third_col: str = "F3"):
        self._rolled = roll_EOM_dynamic_brent(
            self.prices,
            self.expiry_calendar,
            front_col=self.front_col,
            next_col=self.next_col,
            third_col=third_col,
        )
        return self._rolled

    def equity(
        self,
        roll_window: int = 5,
        t_cost: float = 0.01,
        *,
        style: str = "window",
        third_col: str = "F3",
        mid_col: str = "F3",
        far_col: str = "F4",
    ):
        if style == "window":
            self.pnl(roll_window=roll_window)
        elif style == "eom_mid":
            self.pnl_eom_midmonth()
        elif style == "eom_ngl":
            self.pnl_eom_ngl(mid_col=mid_col, far_col=far_col)
        elif style == "eom_eom":
            self.pnl_eom_eom(third_col=third_col)
        elif style == "eom_dynamic":
            self.pnl_eom_dynamic(third_col=third_col)
        else:
            raise ValueError(f"Unknown style '{style}'.")

        self._equity = roll_EL(
            self._rolled,
            self.prices,
            front_col=self.front_col,
            t_cost=t_cost,
        )
        return self._equity

    def metrics(self, contracts=1, units=1000):
        if self._equity is None:
            raise ValueError("Must call .equity() before .metrics().")
        return metrics(self._equity, contracts=contracts, units=units)
