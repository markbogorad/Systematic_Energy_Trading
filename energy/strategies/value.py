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
    t_cost: float = 0.01,          # absolute cost in ORIGINAL quote units
    pct_t_cost: float | None = None,  # fraction of |daily_pnl|, e.g. 0.001 = 10 bps
) -> pd.DataFrame:
    """
    Single-leg value strategy:
      - Signal is based on % deviation of price from its long-run mean.
      - epsilon_factor is interpreted as a *percentage* band around fair value.

    Transaction costs:
      - If pct_t_cost is None or 0: use absolute t_cost (scaled by
        prices.attrs['norm_scale'] into normalized $/bbl units).
      - If pct_t_cost > 0: ignore t_cost and charge pct_t_cost * |daily_pnl|
        per 'turn' (0,1,2) on top of underlying roll costs.
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

    # ---------- PnL / t-cost engine ----------
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
    combined = np.maximum(pnl_df["sig_cost_mult"], roll_mult)

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

    # --- ABSOLUTE vs PERCENT T-COST (with normalization) ---
    norm_scale = prices.attrs.get("norm_scale", 1.0)

    if pct_t_cost is not None and pct_t_cost > 0:
        base_cost = pct_t_cost * pnl_df["daily_pnl"].abs()
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * base_cost
    else:
        abs_tc = t_cost * norm_scale
        pnl_df["sig_t_cost"] = pnl_df["total_cost_mult"] * abs_tc

    # transaction costs: strategy turns + underlying roll costs
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
    epsilon_factor: float = 0.05,       # % deviation band (e.g. 0.05 = 5%)
    t_cost: float = 0.01,               # absolute cost per unit volume traded
    pct_t_cost: float | None = None,    # fraction of |daily_pnl| per unit traded
    total_capital: float = 10_000_000.0,
    pure_spread_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Stat-arb portfolio engine.

    Signals:
      - From pure_spread_prices (true futures levels).
      - epsilon_factor is % deviation from long-run mean.

    Execution:
      - PnL & roll costs from rolled_legs[leg]['daily_pnl', 't_cost'].
      - t_cost / pct_t_cost applied on changes in position volume:

        * If pct_t_cost is None/0: leg_trade_cost = trade_volume * t_cost.
        * If pct_t_cost > 0: base_cost_per_unit = pct_t_cost * |daily_pnl_leg|;
          leg_trade_cost = trade_volume * base_cost_per_unit.
    """

    # ----------------------------
    # 0) Validity checks
    # ----------------------------
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty.")
    if pure_spread_prices is None:
        raise ValueError(
            "pure_spread_prices MUST be provided (never use rolled anchors for signals)."
        )
    if long_ma <= 0:
        raise ValueError("long_ma must be positive.")

    # ----------------------------
    # 1) Capital allocation
    # ----------------------------
    n_pairs = len(pairs)
    notional_per_spread = total_capital / n_pairs
    notional_per_leg    = 0.5 * notional_per_spread

    # ----------------------------
    # 2) BUILD SPREADS FROM PURE PRICE LEVELS
    # ----------------------------
    spreads = _make_spreads_from_prices(pure_spread_prices, pairs)
    idx = spreads.index

    # ----------------------------
    # 3) Long-run mean (expanding -> rolling cap)
    # ----------------------------
    min_obs_ma = min(252, long_ma)

    ma_roll = spreads.rolling(window=long_ma, min_periods=min_obs_ma).mean()
    ma_exp  = spreads.expanding(min_periods=min_obs_ma).mean()
    ma_long = ma_roll.combine_first(ma_exp)

    # ----------------------------
    # 4) % deviation from mean
    # ----------------------------
    d = spreads - ma_long                         # $ deviation
    denom = ma_long.abs().clip(lower=1.0)         # avoid divide-by-zero
    rel_dev = d / denom                           # unitless: 0.05 = 5%

    valid = (~ma_long.isna()) & (~denom.isna())

    # ----------------------------
    # 5) Signals {-1, 0, +1} using % band
    # ----------------------------
    spread_signals = pd.DataFrame(0, index=idx, columns=spreads.columns)

    spread_signals[rel_dev >  epsilon_factor] = -1   # spread rich -> short
    spread_signals[rel_dev < -epsilon_factor] = +1   # spread cheap -> long
    spread_signals[~valid]                    = 0

    # ----------------------------
    # 6) Convert spread signals → leg positions (barrels)
    # ----------------------------
    leg_pos = {leg: pd.Series(0.0, index=idx) for leg in rolled_legs.keys()}

    for (c_i, c_j), spr_name in zip(pairs, spreads.columns):
        sig = spread_signals[spr_name]  # -1, 0, +1

        # Anchor prices for sizing
        px_i = prices[c_i].reindex(idx).ffill()
        px_j = prices[c_j].reindex(idx).ffill()

        qty_i = (notional_per_leg / px_i).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        qty_j = (notional_per_leg / px_j).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # If signal = +1: long i, short j
        leg_pos[c_i] += sig * qty_i
        leg_pos[c_j] -= sig * qty_j

    # ----------------------------
    # 7) EXECUTION on rolled legs
    # ----------------------------
    all_index = None
    for df in rolled_legs.values():
        all_index = df.index if all_index is None else all_index.union(df.index)
    all_index = all_index.union(idx).sort_values()

    total_daily_pnl   = pd.Series(0.0, index=all_index)
    total_t_cost      = pd.Series(0.0, index=all_index)
    any_roll_flag     = pd.Series(0,   index=all_index)
    total_trade_count = pd.Series(0.0, index=all_index)
    port_pos          = pd.Series(0.0, index=all_index)

    for leg, base_df in rolled_legs.items():
        df = base_df[["daily_pnl", "t_cost", "roll_day_flag"]].reindex(all_index).fillna(0)

        pos     = leg_pos[leg].reindex(all_index).ffill().fillna(0.0)
        pos_lag = pos.shift(1).fillna(0.0)

        # Execution PnL
        leg_raw       = pos_lag * df["daily_pnl"]
        leg_roll_cost = df["t_cost"] * pos_lag.abs()

        # Trade volume (absolute change in position)
        trade_volume = (pos - pos_lag).abs()

        # Stat-arb trade cost on volume
        if pct_t_cost is not None and pct_t_cost > 0:
            base_tc = pct_t_cost * df["daily_pnl"].abs()
            leg_trade_cost = trade_volume * base_tc
        else:
            leg_trade_cost = trade_volume * t_cost

        # Trade counting
        enters = (pos_lag == 0.0) & (pos != 0.0)
        exits  = (pos_lag != 0.0) & (pos == 0.0)
        flips  = (pos_lag * pos < 0.0)
        leg_trades = enters.astype(int) + exits.astype(int) + flips.astype(int)

        total_daily_pnl   += leg_raw
        total_t_cost      += (leg_roll_cost + leg_trade_cost)
        any_roll_flag      = np.maximum(any_roll_flag, df["roll_day_flag"])
        total_trade_count += leg_trades
        port_pos          += pos

    # ----------------------------
    # 8) Output
    # ----------------------------
    net_pnl     = total_daily_pnl - total_t_cost
    equity_line = net_pnl.cumsum()
    signal      = np.sign(port_pos).replace(np.nan, 0.0)

    return pd.DataFrame(
        {
            "daily_pnl":   total_daily_pnl,
            "t_cost":      total_t_cost,
            "net_pnl":     net_pnl,
            "roll_flag":   any_roll_flag,
            "equity_line": equity_line,
            "signal":      signal,
            "trade_count": total_trade_count,
        },
        index=all_index,
    )
