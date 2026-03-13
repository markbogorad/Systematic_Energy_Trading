# energy/accounting/mtm.py

from __future__ import annotations

import numpy as np
import pandas as pd

from energy.strategies.rolling import RollingStrategy

try:
    from energy.accounting.contract_specs import CONTRACT_SPECS
except Exception:
    CONTRACT_SPECS = {}


# =============================================================================
# Helpers
# =============================================================================
def _require_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_contract_spec(commodity_name: str) -> dict:
    """
    Expected CONTRACT_SPECS shape:

    CONTRACT_SPECS = {
        "WTI": {
            "ticker": "CL",
            "contract_multiplier": 1000,
            "t_cost_abs": 0.0,
            "normalization": 1.0,
        },
        ...
    }
    """
    if commodity_name not in CONTRACT_SPECS:
        raise KeyError(
            f"Commodity '{commodity_name}' not found in CONTRACT_SPECS. "
            "Either add it there or pass contract_multiplier / t_cost_abs explicitly."
        )

    spec = CONTRACT_SPECS[commodity_name]
    if "contract_multiplier" not in spec:
        raise KeyError(
            f"Commodity '{commodity_name}' missing 'contract_multiplier' in CONTRACT_SPECS."
        )
    return spec


def build_roll_path(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    style: str = "window",
    roll_window: int = 5,
    front_col: str = "F1",
    next_col: str = "F2",
    third_col: str = "F3",
    mid_col: str = "F3",
    far_col: str = "F4",
) -> pd.DataFrame:
    """
    Build the raw rolling path using the trusted RollingStrategy code.

    Returns a DataFrame with, at minimum:
        - daily_pnl
        - held_contract
        - roll_day_flag
    """
    rs = RollingStrategy(
        prices=prices,
        expiry_calendar=expiry_calendar,
        front_col=front_col,
        next_col=next_col,
    )

    if style == "window":
        out = rs.pnl(roll_window=roll_window)
    elif style == "eom_mid":
        out = rs.pnl_eom_midmonth()
    elif style == "eom_ngl":
        out = rs.pnl_eom_ngl(mid_col=mid_col, far_col=far_col)
    elif style == "eom_eom":
        out = rs.pnl_eom_eom(next_col=next_col, third_col=third_col)
    elif style == "eom_dynamic":
        out = rs.pnl_eom_dynamic(third_col=third_col)
    else:
        raise ValueError(f"Unknown style '{style}'.")

    _require_cols(out, ["daily_pnl", "held_contract", "roll_day_flag"])
    return out


def build_held_price_series(
    path_df: pd.DataFrame,
    prices: pd.DataFrame,
    held_col: str = "held_contract",
    output_name: str = "held_price",
) -> pd.Series:
    """
    Map path_df[held_col] back into the raw prices matrix.

    This returns the price level of the contract considered held at each date.
    That series is used for sizing on entry / rebalance dates.
    """
    _require_cols(path_df, [held_col])

    out = pd.Series(index=path_df.index, dtype=float, name=output_name)

    for dt in path_df.index:
        contract = path_df.at[dt, held_col]

        if pd.isna(contract):
            out.at[dt] = np.nan
            continue

        contract = str(contract)
        if contract not in prices.columns:
            raise KeyError(
                f"Held contract '{contract}' on {dt} not found in prices columns."
            )

        if dt not in prices.index:
            raise KeyError(f"Date {dt} not found in prices index.")

        out.at[dt] = float(prices.at[dt, contract])

    return out


# =============================================================================
# Generic MTM engine
# =============================================================================
def mtm_from_path(
    path_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    initial_capital: float,
    contract_multiplier: float,
    pnl_col: str = "daily_pnl",
    held_col: str = "held_contract",
    rebalance_col: str = "roll_day_flag",
    rebalance_mode: str = "signal",
    t_cost_abs: float = 0.0,
    include_exit_cost: bool = False,
    round_contracts: bool = False,
    output_prefix: str = "",
) -> pd.DataFrame:
    """
    Generic MTM accounting engine.

    Parameters
    ----------
    path_df : pd.DataFrame
        Strategy path DataFrame. Must contain at least:
            - pnl_col
            - held_col
            - rebalance_col
    prices : pd.DataFrame
        Raw futures price panel used to recover held prices for sizing.
    initial_capital : float
        Starting capital.
    contract_multiplier : float
        Dollar PnL per 1.0 move in price for one contract.
    pnl_col : str
        Column containing per-1-contract unit pnl.
    held_col : str
        Column containing held contract label (e.g. F1/F2/F3).
    rebalance_col : str
        Column marking rebalance events.
    rebalance_mode : str
        Supported:
            - "signal"    : rebalance when rebalance_col == 1
            - "daily"     : rebalance every day
            - "never"     : size once at inception, never resize after
    t_cost_abs : float
        Absolute cost per contract traded.
    include_exit_cost : bool
        Whether to apply a terminal exit cost on the last date.
    round_contracts : bool
        If True, round to nearest whole contract.
    output_prefix : str
        Optional prefix for MTM output columns if you later want multiple ledgers.

    Returns
    -------
    pd.DataFrame
        Original path_df plus MTM ledger columns.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0.")
    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be > 0.")

    _require_cols(path_df, [pnl_col, held_col, rebalance_col])

    out = path_df.copy().sort_index()
    n = len(out)

    col_held_price = f"{output_prefix}held_price"
    col_contracts = f"{output_prefix}contracts_held"
    col_trade_count = f"{output_prefix}trade_count_mtm"
    col_dollar_pnl = f"{output_prefix}dollar_pnl"
    col_txn_cost = f"{output_prefix}txn_cost_mtm"
    col_capital = f"{output_prefix}capital"
    col_gross = f"{output_prefix}gross_exposure"
    col_rebalance = f"{output_prefix}rebalance_flag"
    col_daily_ret = f"{output_prefix}daily_ret"
    col_equity = f"{output_prefix}equity_index"

    if n == 0:
        out[col_held_price] = pd.Series(dtype=float)
        out[col_contracts] = pd.Series(dtype=float)
        out[col_trade_count] = pd.Series(dtype=float)
        out[col_dollar_pnl] = pd.Series(dtype=float)
        out[col_txn_cost] = pd.Series(dtype=float)
        out[col_capital] = pd.Series(dtype=float)
        out[col_gross] = pd.Series(dtype=float)
        out[col_rebalance] = pd.Series(dtype=float)
        out[col_daily_ret] = pd.Series(dtype=float)
        out[col_equity] = pd.Series(dtype=float)
        return out

    held_price = build_held_price_series(
        out,
        prices,
        held_col=held_col,
        output_name=col_held_price,
    )
    unit_pnl = out[pnl_col].astype(float).to_numpy()
    rebalance_signal = out[rebalance_col].fillna(0).to_numpy(int)

    contracts = np.full(n, np.nan, dtype=float)
    trade_count = np.zeros(n, dtype=float)
    dollar_pnl = np.zeros(n, dtype=float)
    txn_cost = np.zeros(n, dtype=float)
    capital = np.full(n, np.nan, dtype=float)
    gross_exposure = np.full(n, np.nan, dtype=float)
    rebalance_flag = np.zeros(n, dtype=int)

    # -------------------------------------------------------------------------
    # Initial sizing
    # -------------------------------------------------------------------------
    px0 = float(held_price.iloc[0])
    if np.isnan(px0) or px0 == 0:
        raise ValueError("First held price is NaN or zero; cannot initialize MTM account.")

    raw_contracts0 = initial_capital / (px0 * contract_multiplier)
    contracts0 = np.round(raw_contracts0) if round_contracts else raw_contracts0

    if contracts0 == 0:
        raise ValueError(
            "Initial sizing produced 0 contracts. Increase initial_capital or disable rounding."
        )

    entry_turnover = abs(contracts0)
    entry_cost = entry_turnover * abs(t_cost_abs)

    capital0 = initial_capital - entry_cost
    if capital0 <= 0:
        raise ValueError("Initial capital exhausted by entry costs.")

    if not round_contracts:
        contracts0 = capital0 / (px0 * contract_multiplier)

    contracts[0] = contracts0
    trade_count[0] = entry_turnover
    txn_cost[0] = entry_cost
    capital[0] = capital0
    gross_exposure[0] = contracts0 * px0 * contract_multiplier
    rebalance_flag[0] = 1

    # -------------------------------------------------------------------------
    # Daily loop
    # -------------------------------------------------------------------------
    for t in range(1, n):
        prev_contracts = contracts[t - 1]

        # PnL always comes from yesterday's held size
        dollar_pnl[t] = unit_pnl[t] * contract_multiplier * prev_contracts
        capital_pre_rebal = capital[t - 1] + dollar_pnl[t]

        if rebalance_mode == "signal":
            do_rebalance = bool(rebalance_signal[t] == 1)
        elif rebalance_mode == "daily":
            do_rebalance = True
        elif rebalance_mode == "never":
            do_rebalance = False
        else:
            raise ValueError(
                f"Unknown rebalance_mode '{rebalance_mode}'. "
                "Use one of: 'signal', 'daily', 'never'."
            )

        if do_rebalance:
            px_t = float(held_price.iloc[t])
            if np.isnan(px_t) or px_t == 0:
                raise ValueError(
                    f"Held price invalid on rebalance date {out.index[t]}."
                )

            raw_new_contracts = capital_pre_rebal / (px_t * contract_multiplier)
            new_contracts = np.round(raw_new_contracts) if round_contracts else raw_new_contracts

            contract_turnover = abs(new_contracts - prev_contracts)
            cost_t = contract_turnover * abs(t_cost_abs)

            capital_post_cost = capital_pre_rebal - cost_t
            if capital_post_cost <= 0:
                raise ValueError(
                    f"Capital became non-positive after costs on {out.index[t]}."
                )

            if not round_contracts:
                new_contracts = capital_post_cost / (px_t * contract_multiplier)

            contracts[t] = new_contracts
            trade_count[t] = contract_turnover
            txn_cost[t] = cost_t
            capital[t] = capital_post_cost
            rebalance_flag[t] = 1
        else:
            contracts[t] = prev_contracts
            capital[t] = capital_pre_rebal

        px_t = held_price.iloc[t]
        if pd.notna(px_t):
            gross_exposure[t] = contracts[t] * float(px_t) * contract_multiplier

    if include_exit_cost:
        exit_turnover = abs(contracts[-1])
        exit_cost = exit_turnover * abs(t_cost_abs)
        txn_cost[-1] += exit_cost
        capital[-1] -= exit_cost
        trade_count[-1] += exit_turnover

    capital_series = pd.Series(capital, index=out.index, name=col_capital)

    out[col_held_price] = held_price
    out[col_contracts] = contracts
    out[col_trade_count] = trade_count
    out[col_dollar_pnl] = dollar_pnl
    out[col_txn_cost] = txn_cost
    out[col_capital] = capital_series
    out[col_gross] = gross_exposure
    out[col_rebalance] = rebalance_flag
    out[col_daily_ret] = capital_series.pct_change()
    out[col_equity] = capital_series / float(initial_capital)

    return out


# =============================================================================
# Backward-compatible rolling-specific wrapper
# =============================================================================
def mtm_from_roll_path(
    rolled_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    initial_capital: float,
    contract_multiplier: float,
    rebalance_mode: str = "roll_only",
    t_cost_abs: float = 0.0,
    include_exit_cost: bool = False,
    round_contracts: bool = False,
    held_col: str = "held_contract",
) -> pd.DataFrame:
    """
    Rolling-specific wrapper around mtm_from_path().

    Uses:
        pnl_col       = 'daily_pnl'
        held_col      = held_col
        rebalance_col = 'roll_day_flag'
    """
    if rebalance_mode == "roll_only":
        generic_rebalance_mode = "signal"
    elif rebalance_mode in {"daily", "never"}:
        generic_rebalance_mode = rebalance_mode
    else:
        raise ValueError(
            f"Unknown rebalance_mode '{rebalance_mode}'. "
            "Use one of: 'roll_only', 'daily', 'never'."
        )

    return mtm_from_path(
        path_df=rolled_df,
        prices=prices,
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
        pnl_col="daily_pnl",
        held_col=held_col,
        rebalance_col="roll_day_flag",
        rebalance_mode=generic_rebalance_mode,
        t_cost_abs=t_cost_abs,
        include_exit_cost=include_exit_cost,
        round_contracts=round_contracts,
        output_prefix="",
    )


# =============================================================================
# High-level rolling convenience wrapper
# =============================================================================
def mtm_roll_account(
    prices: pd.DataFrame,
    expiry_calendar: pd.DatetimeIndex,
    *,
    initial_capital: float,
    commodity_name: str | None = None,
    contract_multiplier: float | None = None,
    t_cost_abs: float | None = None,
    style: str = "window",
    roll_window: int = 5,
    front_col: str = "F1",
    next_col: str = "F2",
    third_col: str = "F3",
    mid_col: str = "F3",
    far_col: str = "F4",
    rebalance_mode: str = "roll_only",
    include_exit_cost: bool = False,
    round_contracts: bool = False,
) -> pd.DataFrame:
    """
    End-to-end MTM builder for long rolling futures.

    This:
      1) builds the trusted roll path using RollingStrategy
      2) constructs a true MTM capital account from that path

    You can specify either:
      - commodity_name (resolve from CONTRACT_SPECS), or
      - contract_multiplier directly
    """
    if contract_multiplier is None:
        if commodity_name is None:
            raise ValueError(
                "Must provide either contract_multiplier directly or commodity_name "
                "resolvable via CONTRACT_SPECS."
            )
        spec = get_contract_spec(commodity_name)
        contract_multiplier = float(spec["contract_multiplier"])
        if t_cost_abs is None:
            t_cost_abs = float(spec.get("t_cost_abs", 0.0))
    else:
        contract_multiplier = float(contract_multiplier)
        if t_cost_abs is None:
            t_cost_abs = 0.0

    rolled_df = build_roll_path(
        prices=prices,
        expiry_calendar=expiry_calendar,
        style=style,
        roll_window=roll_window,
        front_col=front_col,
        next_col=next_col,
        third_col=third_col,
        mid_col=mid_col,
        far_col=far_col,
    )

    out = mtm_from_roll_path(
        rolled_df=rolled_df,
        prices=prices,
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
        rebalance_mode=rebalance_mode,
        t_cost_abs=float(t_cost_abs),
        include_exit_cost=include_exit_cost,
        round_contracts=round_contracts,
        held_col="held_contract",
    )

    if commodity_name is not None:
        out["commodity"] = commodity_name

    out["contract_multiplier"] = float(contract_multiplier)
    out["initial_capital"] = float(initial_capital)
    out["rebalance_mode"] = rebalance_mode

    return out