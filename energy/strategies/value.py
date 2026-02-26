import pandas as pd
import numpy as np
from collections.abc import Iterable

def value(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F1",
    long_ma: int = 252 * 5,        # target mean-reversion horizon
    vol_window: int | None = None, # kept for compatibility (ignored)
    epsilon_factor: float = 0.05,  # % deviation threshold (e.g. 0.05 = 5%)
    t_cost: float = 0.00,          # absolute cost in ORIGINAL quote units
    pct_t_cost: float | None = None,  # fraction of price, e.g. 0.001 = 10 bps
) -> pd.DataFrame:
    """
    Single-leg value strategy:
      - Signal is based on % deviation of price from its long-run mean.
      - epsilon_factor is interpreted as a *percentage* band around fair value.

    Transaction costs:
      - If pct_t_cost is None or 0: use absolute t_cost (scaled by
        prices.attrs['norm_scale'] into normalized $/bbl units).
      - If pct_t_cost > 0: cost per 'turn' is pct_t_cost * |price|
        on top of underlying roll costs (from rolled_df['t_cost']).
    """

    df = prices.copy()
    px = df[front_col].astype(float)

    # ---------- 1) Long-run mean: expanding -> rolling cap ----------
    if long_ma <= 0:
        raise ValueError("long_ma must be positive.")

    min_obs_ma = min(252, long_ma)  # minimum obs for MA

    ma_roll = px.rolling(window=long_ma, min_periods=min_obs_ma).mean()
    ma_exp  = px.expanding(min_periods=min_obs_ma).mean()
    ma_long = ma_roll.combine_first(ma_exp)

    # ---------- 2) % deviation from mean ----------
    d = px - ma_long                                   # $ deviation
    denom = ma_long.abs().clip(lower=1.0)             # avoid divide-by-zero
    rel_dev = d / denom                               # unitless: 0.05 = 5%

    # ---------- Start date: once MA is defined ----------
    valid = (~ma_long.isna()) & (~denom.isna())
    if valid.any():
        start_date = valid.idxmax()
    else:
        start_date = df.index[-1]  # degenerate; remain flat

    # ---------- 3) Discretize to {-1,0,+1} based on % band ----------
    sig = pd.Series(0, index=df.index)
    rich  = rel_dev >  epsilon_factor   # price >> fair -> short
    cheap = rel_dev < -epsilon_factor   # price << fair -> long
    sig[rich]  = -1
    sig[cheap] = +1
    df["signal"] = sig

    # ---------- 4) PnL / t-cost engine ----------
    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()

    # align signal to rolled_df; trade with lagged signal
    pnl_df["signal"]     = df["signal"].reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # no positions / no extra roll costs before warm-up
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    # raw value pnl
    pnl_df["val_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # --- signal-change cost multiplier ---
    delta = (pnl_df["signal"] - pnl_df["signal_lag"]).abs()
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta >= 2],
        [0, 1, 2],
        default=0,
    )

    # --- roll multiplier ---
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined  = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

    # flat-overlap edge-case: exiting to flat on roll day
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined[flat_overlap] = 1

    pnl_df["total_cost_mult"] = combined

    # force at least one turn first/last bar (defensive)
    if not pnl_df.empty:
        i0, i1 = pnl_df.index[0], pnl_df.index[-1]
        pnl_df.loc[i0, "total_cost_mult"] = max(
            pnl_df.loc[i0, "total_cost_mult"], 1
        )
        pnl_df.loc[i1, "total_cost_mult"] = max(
            pnl_df.loc[i1, "total_cost_mult"], 1
        )

    # cap at 2 "turns" per day
    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)

    # --- trade count = total number of turns that day (0, 1, or 2) ---
    pnl_df["trade_count"] = pnl_df["total_cost_mult"]

    # --- ABSOLUTE vs PERCENT T-COST (on PRICE) ---
    norm_scale = prices.attrs.get("norm_scale", 1.0)
    px_for_cost = prices[front_col].reindex(pnl_df.index).astype(float)

    if pct_t_cost is not None and pct_t_cost > 0:
        base_cost = pct_t_cost * px_for_cost.abs()
    else:
        abs_tc    = t_cost * norm_scale
        base_cost = pd.Series(abs_tc, index=pnl_df.index, dtype=float)

    sig_cost = pnl_df["total_cost_mult"] * base_cost

    # roll costs from underlying roll engine (negative in rolled_df)
    roll_t_cost = rolled_df["t_cost"].reindex(pnl_df.index).fillna(0.0)
    roll_cost   = -roll_t_cost   # positive

    total_cost = sig_cost + roll_cost

    pnl_df["t_cost"] = total_cost
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0.0

    # net PnL & equity
    pnl_df["net_pnl"]     = pnl_df["val_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    out = pnl_df.rename(
    columns={"mom_raw": "daily_pnl", "roll_day_flag": "roll_flag"}
    )
    # make sure signal is float and included in the output
    out["signal"] = out["signal"].astype(float)

    return out[[
        "daily_pnl",
        "t_cost",
        "net_pnl",
        "roll_flag",
        "equity_line",
        "trade_count",
        "signal",
    ]]



def _make_spreads_from_prices(
    prices: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    prefix: str = "spr",
) -> pd.DataFrame:
    """
    Build level spreads S_ij = P_i - P_j from TRUE PRICE LEVELS.

    This must be passed pure curve prices (F12/F13 or whichever tenor you choose).
    Rolling anchors should NEVER be used here.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty (col_i, col_j).")

    spreads = {}
    for (c_i, c_j) in pairs:
        if c_i not in prices.columns or c_j not in prices.columns:
            raise KeyError(f"Missing {c_i} or {c_j} in pure_spread_prices.")
        name = f"{prefix}_{c_i}_minus_{c_j}"
        spreads[name] = prices[c_i].astype(float) - prices[c_j].astype(float)

    return pd.DataFrame(spreads)

def statistical_arbitrage(
    rolled_legs: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    long_ma: int = 252 * 5,
    vol_window: int | None = None,      # kept for signature compat (ignored for band)
    epsilon_factor: float = 0.00,       # % deviation band (e.g. 0.05 = 5%)
    t_cost: float = 0.00,               # absolute cost per unit volume traded
    pct_t_cost: float | None = None,    # fraction of PRICE per unit traded
    total_capital: float = 10_000_000.0,
    pure_spread_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Stat-arb engine, NO-lookahead convention:
      - compute signal at t from prices through t
      - position held on day t is based on signal at t-1  (shift(1))
      - PnL at t uses position held at t: pos[t] * daily_pnl[t]
      - trade costs at t use position change: |pos[t] - pos[t-1]|
    """

    # ----------------------------
    # 0) Validity checks
    # ----------------------------
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty.")
    if pure_spread_prices is None:
        raise ValueError("pure_spread_prices MUST be provided (never use rolled anchors for signals).")
    if long_ma <= 0:
        raise ValueError("long_ma must be positive.")
    if not rolled_legs:
        raise ValueError("rolled_legs must be non-empty.")

    missing_price_cols = [k for k in rolled_legs.keys() if k not in prices.columns]
    if missing_price_cols:
        raise KeyError(f"Missing price columns for legs: {missing_price_cols}")

    # ----------------------------
    # 1) Capital allocation
    # ----------------------------
    n_pairs = len(pairs)
    notional_per_spread = float(total_capital) / n_pairs
    notional_per_leg = 0.5 * notional_per_spread

    # ----------------------------
    # 2) Build spreads from PURE levels
    # ----------------------------
    spreads = _make_spreads_from_prices(pure_spread_prices, pairs).sort_index()
    idx = spreads.index
    if idx.size < 2:
        raise RuntimeError("Spread index too short to run stat-arb.")

    # ----------------------------
    # 3) Long-run mean (expanding -> rolling cap)
    # ----------------------------
    min_obs_ma = min(252, long_ma)
    ma_roll = spreads.rolling(window=long_ma, min_periods=min_obs_ma).mean()
    ma_exp  = spreads.expanding(min_periods=min_obs_ma).mean()
    ma_long = ma_roll.combine_first(ma_exp)

    # ----------------------------
    # 4) % deviation from mean  (NOTE: your clip(lower=1.0) breaks scale invariance near 0)
    # ----------------------------
    d = spreads - ma_long
    denom = ma_long.abs().clip(lower=1.0)
    rel_dev = d / denom
    valid = (~ma_long.isna()) & (~denom.isna())

    # ----------------------------
    # 5) Spread band signals {-1, 0, +1}
    # ----------------------------
    spread_signals = pd.DataFrame(0.0, index=idx, columns=spreads.columns)
    spread_signals[rel_dev >  epsilon_factor] = -1.0
    spread_signals[rel_dev < -epsilon_factor] = +1.0
    spread_signals[~valid] = 0.0

    # ----------------------------
    # 6) Spread signals -> leg positions on idx (then SHIFT(1) for no-lookahead)
    # ----------------------------
    leg_pos = {leg: pd.Series(0.0, index=idx, dtype=float) for leg in rolled_legs.keys()}

    for (c_i, c_j), spr_name in zip(pairs, spreads.columns):
        if c_i not in rolled_legs or c_j not in rolled_legs:
            raise KeyError(f"Pair leg missing in rolled_legs: ({c_i}, {c_j})")

        sig = spread_signals[spr_name].astype(float)

        px_i = prices[c_i].reindex(idx).ffill()
        px_j = prices[c_j].reindex(idx).ffill()
        if px_i.isna().any() or px_j.isna().any():
            raise ValueError(f"NaNs in sizing prices after ffill for pair ({c_i}, {c_j}).")

        qty_i = (notional_per_leg / px_i).replace([np.inf, -np.inf], np.nan)
        qty_j = (notional_per_leg / px_j).replace([np.inf, -np.inf], np.nan)
        if qty_i.isna().any() or qty_j.isna().any():
            raise ValueError(f"NaNs/inf in computed quantities for pair ({c_i}, {c_j}).")

        leg_pos[c_i] = leg_pos[c_i] + sig * qty_i
        leg_pos[c_j] = leg_pos[c_j] - sig * qty_j

    leg_pos_df = pd.DataFrame(leg_pos).sort_index()

    # ---- NO-LOOKAHEAD: trade tomorrow (position on t uses signal from t-1)
    leg_pos_df = leg_pos_df.shift(1)

    # leg signal after shift (for attrs)
    leg_sig_df = np.sign(leg_pos_df).fillna(0.0).astype(float)

    # ----------------------------
    # 7) STRICT execution calendar = intersection of idx and all rolled legs
    # ----------------------------
    exec_idx = idx.copy()
    for leg, df in rolled_legs.items():
        exec_idx = exec_idx.intersection(df.index)
    exec_idx = exec_idx.sort_values()

    if exec_idx.size < 2:
        raise RuntimeError("Execution index too short after strict intersection across legs + spreads.")

    # positions aligned to execution calendar
    leg_pos_exec = leg_pos_df.reindex(exec_idx)

    # we cannot trade until positions are defined (post shift + ffill)
    leg_pos_exec = leg_pos_exec.ffill()
    good_mask = ~leg_pos_exec.isna().any(axis=1)
    exec_idx = exec_idx[good_mask.values]
    leg_pos_exec = leg_pos_exec.loc[exec_idx]

    if exec_idx.size < 2:
        raise RuntimeError("Execution index too short after dropping NaN-position warmup dates.")

    # ----------------------------
    # 8) Execute strictly
    # ----------------------------
    total_daily_pnl   = pd.Series(0.0, index=exec_idx, dtype=float)
    total_t_cost      = pd.Series(0.0, index=exec_idx, dtype=float)
    any_roll_flag     = pd.Series(0,   index=exec_idx, dtype=int)
    total_trade_count = pd.Series(0.0, index=exec_idx, dtype=float)
    port_pos          = pd.Series(0.0, index=exec_idx, dtype=float)

    for leg, base_df in rolled_legs.items():
        req_cols = ["daily_pnl", "t_cost", "roll_day_flag"]
        missing_cols = [c for c in req_cols if c not in base_df.columns]
        if missing_cols:
            raise ValueError(f"[{leg}] missing required cols: {missing_cols}")

        df = base_df[req_cols].reindex(exec_idx)
        if df.isna().any().any():
            bad = df.isna().any(axis=1)
            first_bad = df.index[bad][0]
            raise ValueError(f"[{leg}] NaNs in execution inputs on exec_idx (first at {first_bad}).")

        pos = leg_pos_exec[leg].astype(float)
        pos_lag = pos.shift(1)

        # PnL at t uses held position at t (already lagged from signal): NO lookahead
        leg_raw = pos * df["daily_pnl"].astype(float)

        # Roll cost should apply to the held position at t (same convention)
        roll_tc_per_unit = -df["t_cost"].astype(float)
        leg_roll_cost = roll_tc_per_unit * pos.abs()

        # Trade costs on turnover at t
        trade_volume = (pos - pos_lag).abs()

        if pct_t_cost is not None and pct_t_cost > 0:
            px_leg = prices[leg].reindex(exec_idx).ffill()
            if px_leg.isna().any():
                bad = px_leg.isna()
                first_bad = px_leg.index[bad][0]
                raise ValueError(f"[{leg}] NaNs in prices for pct_t_cost (first at {first_bad}).")
            base_tc = float(pct_t_cost) * px_leg.abs()
            leg_trade_cost = trade_volume * base_tc
        else:
            leg_trade_cost = trade_volume * float(t_cost)

        leg_total_cost = leg_roll_cost + leg_trade_cost

        # Trade counting (based on turnover)
        enters = (pos_lag == 0.0) & (pos != 0.0)
        exits  = (pos_lag != 0.0) & (pos == 0.0)
        flips  = (pos_lag * pos < 0.0)
        leg_trades = enters.astype(int) + exits.astype(int) + flips.astype(int)

        total_daily_pnl   = total_daily_pnl + leg_raw
        total_t_cost      = total_t_cost + leg_total_cost
        any_roll_flag     = np.maximum(any_roll_flag, df["roll_day_flag"].astype(int))
        total_trade_count = total_trade_count + leg_trades.astype(float)
        port_pos          = port_pos + pos

    # drop first day where pos_lag is NaN (and any unexpected NaNs)
    tmp = pd.DataFrame(
        {
            "daily_pnl": total_daily_pnl,
            "t_cost": total_t_cost,
            "roll_flag": any_roll_flag,
            "trade_count": total_trade_count,
            "port_pos": port_pos,
        },
        index=exec_idx,
    ).replace([np.inf, -np.inf], np.nan).dropna(how="any")

    exec_idx2 = tmp.index
    if exec_idx2.size < 2:
        raise RuntimeError("Too few execution rows after dropping NaNs.")

    # ----------------------------
    # 9) Portfolio PnL / equity
    # ----------------------------
    net_pnl = (tmp["daily_pnl"] - tmp["t_cost"]).astype(float)
    equity_line = net_pnl.cumsum()

    # ----------------------------
    # 10) Portfolio-level spread signal aligned to exec_idx2
    #     (signal at t is spread_sig[t-1] if you want reporting consistent with positions)
    # ----------------------------
    if spread_signals.shape[1] == 1:
        spread_sig = spread_signals.iloc[:, 0].astype(float)
    else:
        spread_sig = spread_signals.mean(axis=1).astype(float)

    # since positions are shifted(1), shift the portfolio signal too for consistency
    signal = spread_sig.shift(1).reindex(exec_idx2).ffill().fillna(0.0).clip(-1.0, 1.0)

    # Leg signals aligned to exec_idx2
    leg_sig_full = leg_sig_df.reindex(exec_idx2).ffill().fillna(0.0).clip(-1.0, 1.0)

    result = pd.DataFrame(
        {
            "daily_pnl":   tmp["daily_pnl"].astype(float),
            "t_cost":      tmp["t_cost"].astype(float),
            "net_pnl":     net_pnl.astype(float),
            "roll_flag":   tmp["roll_flag"].astype(int),
            "equity_line": equity_line.astype(float),
            "signal":      signal.astype(float),
            "trade_count": tmp["trade_count"].astype(float),
        },
        index=exec_idx2,
    )

    result.attrs["spread_signals"] = spread_signals
    result.attrs["leg_signals"] = leg_sig_full
    return result


def carry_value(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    front_col: str = "F4",          # nearer contract, e.g. F1
    end_col: str = "F15",            # farther contract, e.g. F4
    long_ma: int = 252 * 5,         # horizon for "fair" timespread MA_z
    vol_window: int | None = None,  # if set, use z-score over this window
    epsilon_factor: float = 0.0,    # dead-zone band on *signal* (0 = pure sign)
    t_cost: float = 0.00,           # abs cost in ORIGINAL quote units
    pct_t_cost: float | None = None # fraction of |daily_pnl|, e.g. 0.001 = 10 bps
) -> pd.DataFrame:
    """
    Curve-based value signal ("carry_value").

    Timespread:
        TS_t = F_front(t) - F_end(t)

    Value signal (mean reversion, non-standardized):
        pi_t = -(TS_t - MA_z(TS_t))

    Optional z-score version (if vol_window is given):
        pi_t = -(TS_t - MA_z(TS_t)) / sigma_z(TS_t)

    Trading rule:
        - We compute a continuous signal `pi_t_cont`.
        - Positions are sign(pi_t_cont) with an optional dead-zone:
              if pi_t_cont >  epsilon_factor -> +1 (long)
              if pi_t_cont < -epsilon_factor -> -1 (short)
              else                           -> 0 (flat)

      Positions are taken in the rolled front contract (same as `value`/`carry`),
      using rolled_df['daily_pnl'] and rolled_df['t_cost'].
    """

    df = prices.copy()

    # --------- 1) Build timespread from explicit contracts ----------
    ts = (df[front_col].astype(float) - df[end_col].astype(float)).rename("timespread")

    if long_ma <= 0:
        raise ValueError("long_ma must be positive.")

    # --------- 2) Long-run mean of timespread MA_z(TS_t) ----------
    min_obs_ma = min(252, long_ma)

    ma_roll = ts.rolling(window=long_ma, min_periods=min_obs_ma).mean()
    ma_exp  = ts.expanding(min_periods=min_obs_ma).mean()
    ma_long = ma_roll.combine_first(ma_exp)

    # deviation from long-run mean
    dev = ts - ma_long   # TS_t - MA_z(TS_t)

    # optional sigma_z(TS_t) for z-score version
    if vol_window is not None and vol_window > 0:
        vol = ts.rolling(window=vol_window, min_periods=min_obs_ma).std()
        vol = vol.replace(0, np.nan)
        pi_cont = -(dev / vol)      # z-score version
    else:
        pi_cont = -dev              # non-standardized version

    # start date: once MA is defined (and vol if used)
    valid = ~ma_long.isna()
    if vol_window is not None and vol_window > 0:
        valid &= ~vol.isna()

    if valid.any():
        start_date = valid.idxmax()
    else:
        start_date = df.index[-1]  # degenerate; remain flat

    # --------- 3) Discretize pi_cont to {-1,0,+1} using epsilon_factor band ----------
    sig = pd.Series(0, index=df.index)

    # note: epsilon_factor is on the *pi_cont* magnitude now, not % deviation
    rich  = pi_cont < -epsilon_factor  # very negative signal -> short
    cheap = pi_cont >  epsilon_factor  # very positive signal -> long

    sig[rich]  = -1
    sig[cheap] = +1

    df["signal_raw"] = pi_cont
    df["signal"]     = sig

    # --------- 4) PnL / t-cost engine (same pattern as value()) ----------
    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()

    pnl_df["signal"]     = df["signal"].reindex(pnl_df.index).ffill().fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # no positions / extra roll costs before warm-up
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    # raw pnl from curve-based value signal
    pnl_df["val_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # --- signal-change cost multiplier ---
    delta = (pnl_df["signal"] - pnl_df["signal_lag"]).abs()
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta >= 2],
        [0, 1, 2],
        default=0,
    )

    # --- roll multiplier ---
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

    # flat-overlap edge-case: exiting to flat on roll day
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined[flat_overlap] = 1

    pnl_df["total_cost_mult"] = combined

    # enforce at least one turn on first/last obs
    if not pnl_df.empty:
        i0, i1 = pnl_df.index[0], pnl_df.index[-1]
        pnl_df.loc[i0, "total_cost_mult"] = max(
            pnl_df.loc[i0, "total_cost_mult"], 1
        )
        pnl_df.loc[i1, "total_cost_mult"] = max(
            pnl_df.loc[i1, "total_cost_mult"], 1
        )

    # cap at 2 turns/day
    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)

    # trade count = total turns that day
    pnl_df["trade_count"] = pnl_df["total_cost_mult"]

    # --- ABSOLUTE vs PERCENT T-COST (with normalization) ---
    norm_scale = prices.attrs.get("norm_scale", 1.0)

    if pct_t_cost is not None and pct_t_cost > 0:
        base_cost = pct_t_cost * pnl_df["daily_pnl"].abs()
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * base_cost
    else:
        abs_tc = t_cost * norm_scale
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * abs_tc

    # add underlying roll costs
    pnl_df["t_cost"] = (
        pnl_df["sig_t_cost"]
        + rolled_df["t_cost"].reindex(pnl_df.index).fillna(0)
    )
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0

    # net PnL & equity
    pnl_df["net_pnl"]     = pnl_df["val_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    return pnl_df.rename(
        columns={"val_raw": "daily_pnl", "roll_day_flag": "roll_flag"}
    )[["daily_pnl", "t_cost", "net_pnl", "roll_flag", "equity_line", "trade_count"]]

def momentum_with_value_filter(
    prices: pd.DataFrame,
    rolled_df: pd.DataFrame,
    mom_signal: pd.Series,
    val_signal: pd.Series,
    t_cost: float = 0.01,          # absolute cost in ORIGINAL quote units
    pct_t_cost: float | None = None  # fraction of |daily_pnl|, e.g. 0.001 = 10 bps
) -> pd.DataFrame:
    """
    Short-term momentum, long-term value signal.

    Economics:
      - Momentum provides the *direction* of the trade.
      - Value acts as a *certainty filter* on the long-run equilibrium.
      - If value is "flat" (value == 0), we do not trade momentum.
      - If value is non-zero, we follow momentum.

    Implementation:
      - mom_signal, val_signal are expected to be in {-1, 0, +1}.
      - Combined signal:

            if val_signal != 0:  use mom_signal
            if val_signal == 0:  go flat

        i.e.  pi_t = 1{val_t != 0} * mom_t
    """

    # ------------------------
    # 0) Align and build signal
    # ------------------------
    idx = rolled_df.index

    sig_df = pd.DataFrame(
        {
            "mom": mom_signal.reindex(idx),
            "val": val_signal.reindex(idx),
        },
        index=idx,
    )

    # determine warm-up start: first date both signals are defined
    valid = (~sig_df["mom"].isna()) & (~sig_df["val"].isna())
    if valid.any():
        # use boolean mask + .loc and take the first timestamp
        start_date = sig_df.loc[valid].index[0]
    else:
        # degenerate case: never trade
        start_date = idx[-1]

    # forward-fill and fill missing with 0 after warm-up
    sig_df = sig_df.ffill().fillna(0)

    # value-as-certainty filter:
    # if val == 0 -> flat; if val != 0 -> follow momentum
    combined_sig = pd.Series(0, index=idx)
    combined_sig[sig_df["val"] != 0] = sig_df["mom"][sig_df["val"] != 0]

    # ------------------------
    # 1) PnL / t-cost engine
    # ------------------------
    pnl_df = rolled_df[["daily_pnl", "t_cost", "roll_day_flag"]].copy()

    pnl_df["signal"]     = combined_sig.reindex(idx).fillna(0)
    pnl_df["signal_lag"] = pnl_df["signal"].shift(1).fillna(0)

    # no positions / extra roll costs before warm-up
    pnl_df.loc[pnl_df.index < start_date, ["signal", "signal_lag"]] = 0
    pnl_df.loc[pnl_df.index < start_date, "roll_day_flag"] = 0

    # raw pnl from filtered momentum signal
    pnl_df["strat_raw"] = pnl_df["signal_lag"] * pnl_df["daily_pnl"]

    # --- signal-change cost multiplier (0,1,2 turns) ---
    delta = (pnl_df["signal"] - pnl_df["signal_lag"]).abs()
    pnl_df["sig_cost_mult"] = np.select(
        [delta == 0, delta == 1, delta >= 2],
        [0, 1, 2],
        default=0,
    )

    # --- roll multiplier (2 turns on roll days) ---
    roll_mult = pnl_df["roll_day_flag"] * 2
    combined_mult = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

    # flat-overlap edge-case: exiting to flat on roll day -> only 1 turn
    flat_overlap = (
        (pnl_df["roll_day_flag"] == 1)
        & (pnl_df["signal_lag"] != 0)
        & (pnl_df["signal"] == 0)
    )
    combined_mult[flat_overlap] = 1

    pnl_df["total_cost_mult"] = combined_mult

    # enforce at least one turn on first/last obs (defensive)
    if not pnl_df.empty:
        i0, i1 = pnl_df.index[0], pnl_df.index[-1]
        pnl_df.loc[i0, "total_cost_mult"] = max(
            pnl_df.loc[i0, "total_cost_mult"], 1
        )
        pnl_df.loc[i1, "total_cost_mult"] = max(
            pnl_df.loc[i1, "total_cost_mult"], 1
        )

    # cap at 2 turns/day
    pnl_df["total_cost_mult"] = pnl_df["total_cost_mult"].clip(upper=2)

    # trade count = total turns that day
    pnl_df["trade_count"] = pnl_df["total_cost_mult"]

    # ------------------------
    # 2) ABSOLUTE vs PERCENT T-COST (with normalization)
    # ------------------------
    norm_scale = prices.attrs.get("norm_scale", 1.0)

    if pct_t_cost is not None and pct_t_cost > 0:
        base_cost = pct_t_cost * pnl_df["daily_pnl"].abs()
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * base_cost
    else:
        abs_tc = t_cost * norm_scale
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * abs_tc

    # add underlying roll costs
    pnl_df["t_cost"] = (
        pnl_df["sig_t_cost"]
        + rolled_df["t_cost"].reindex(idx).fillna(0)
    )
    pnl_df.loc[pnl_df.index < start_date, "t_cost"] = 0

    # net PnL & equity
    pnl_df["net_pnl"]     = pnl_df["strat_raw"] - pnl_df["t_cost"]
    pnl_df["equity_line"] = pnl_df["net_pnl"].cumsum()

    out = pnl_df.rename(
    columns={"mom_raw": "daily_pnl", "roll_day_flag": "roll_flag"}
    )
    # make sure signal is float and included in the output
    out["signal"] = out["signal"].astype(float)

    return out[[
        "daily_pnl",
        "t_cost",
        "net_pnl",
        "roll_flag",
        "equity_line",
        "trade_count",
        "signal",
    ]]


