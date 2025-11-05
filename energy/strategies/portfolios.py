# strategies/portfolios.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, Tuple

# =====================================================================
# Helpers
# =====================================================================

RebalFreq = Literal["W", "M", "Q"]

def _rebal_dates(idx: pd.DatetimeIndex, freq: RebalFreq = "M") -> pd.DatetimeIndex:
    """
    Rebalance dates = last available trading day of each period in `idx`.
    No look-ahead: all dates are members of `idx`.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    tmp = pd.DataFrame(index=idx)
    period_ends = tmp.resample(freq).last().index
    pos = np.searchsorted(idx.values, period_ends.values, side="right") - 1
    pos = pos[(pos >= 0)]
    return idx[pos].unique()

def _subsample(df: pd.DataFrame, end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Window including end_date (inclusive) with length <= lookback."""
    end_loc = df.index.get_loc(end_date)
    start = max(0, end_loc - lookback + 1)
    return df.iloc[start : end_loc + 1]

def _safe_inv(mat: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Numerically safe inverse via Tikhonov ridge."""
    n = mat.shape[0]
    return np.linalg.pinv(mat + ridge * np.eye(n))

def _forward_fill_weights(w_hist: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Expand sparse weight snapshots to daily weights by forward fill.
    If leading NaNs exist (before first rebalance), fill with first available.
    """
    W = w_hist.reindex(full_index).ffill()
    first_row = W.dropna(how="all").iloc[0] if not W.dropna(how="all").empty else None
    if first_row is not None:
        W = W.fillna(first_row)
    else:
        W = W.fillna(0.0)
    return W

def _portfolio_from_weights(pnl: pd.DataFrame, weights_daily: pd.DataFrame) -> Dict[str, pd.Series | pd.DataFrame]:
    """Apply daily weights to daily net PnL columns."""
    pnl = pnl.reindex(weights_daily.index).reindex(columns=weights_daily.columns)
    port_pnl = (weights_daily * pnl).sum(axis=1)
    return {
        "pnl": port_pnl,
        "equity": port_pnl.cumsum(),
        "weights_daily": weights_daily,
    }

def _inv_vol_weights(window: pd.DataFrame, *, long_only: bool = True) -> pd.Series:
    """Inverse-vol weights on the provided window (pairwise complete)."""
    vol = window.std(ddof=0).replace(0, np.nan)
    w = 1.0 / vol.values
    if np.all(~np.isfinite(w)):
        w = np.ones(len(vol))
    w = np.nan_to_num(w, nan=0.0)
    if long_only:
        w = np.clip(w, 0, None)
    s = w.sum()
    w = w / s if s > 0 else np.ones_like(w) / len(w)
    return pd.Series(w, index=window.columns)

def _erc_weights(
    S: np.ndarray,
    w0: np.ndarray,
    *,
    long_only: bool = True,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """
    Equal Risk Contribution weights via multiplicative fixed-point updates.
    Returns (weights, converged_flag).
    """
    w = w0.copy()
    n = len(w)
    for _ in range(max_iter):
        Sw = S @ w
        total = float(w @ Sw)
        if not np.isfinite(total) or total <= 0:
            return w, False
        RC = w * Sw
        target = total / n
        w_new = w * (target / (RC + 1e-16))
        if long_only:
            w_new = np.clip(w_new, 0.0, None)
        s = w_new.sum()
        if s <= 0:
            return w, False
        w_new /= s
        if np.linalg.norm(w_new - w, ord=1) < tol:
            return w_new, True
        w = w_new
    return w, False

def _align_two(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Strict column intersection + aligned index (outer join then drop rows where either side is all-NaN).
    Returns (a_aligned, b_aligned, common_cols).
    """
    a = a.copy().sort_index()
    b = b.copy().sort_index()
    common_cols = sorted(set(a.columns) & set(b.columns))
    a = a[common_cols]
    b = b[common_cols]
    idx = a.index.union(b.index)
    a = a.reindex(idx)
    b = b.reindex(idx)
    return a, b, common_cols

# =====================================================================
# 1) Equal Weight
# =====================================================================

def equal_weight_static(
    pnl: pd.DataFrame,
    *,
    use_all_columns: bool = True,   # if False, uses only columns that ever have data
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Constant equal-weight portfolio (no rebal): w_i = 1/N for all i, all days.
    Assumes a fixed universe defined by pnl.columns (N = number of columns).
    """
    pnl = pnl.copy().sort_index()
    if use_all_columns:
        cols = pnl.columns
    else:
        cols = pnl.loc[:, pnl.notna().any()].columns

    N = len(cols)
    if N == 0:
        raise ValueError("No assets found for equal_weight_static.")

    w0 = pd.Series(1.0 / N, index=cols)
    snap = pd.DataFrame([w0], index=[pnl.index[0]]).reindex(columns=pnl.columns).fillna(0.0)
    W = snap.reindex(pnl.index).ffill().fillna(0.0)

    port_pnl = (W * pnl).sum(axis=1)
    return {"pnl": port_pnl, "equity": port_pnl.cumsum(), "weights_daily": W}

# =====================================================================
# 2) Risk Parity (ERC) — Static and Dynamic (strategy-specific risk)
# =====================================================================

def risk_parity_static(
    pnl: pd.DataFrame,
    *,
    risk_basis: Optional[pd.DataFrame] = None,  # strategy series to estimate risk (std/cov)
    long_only: bool = True,
    ridge: float = 1e-6,
    diag_only: bool = False,                    # if True, ignore cross-covariances
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    STATIC Risk Parity using END-OF-SAMPLE statistics (look-ahead).
    'risk_basis' defines the risk model (e.g., MA or Carry PnL); weights applied to 'pnl'.
    """
    pnl = pnl.copy().sort_index()
    basis = (risk_basis if risk_basis is not None else pnl)
    pnl, basis, common_cols = _align_two(pnl, basis)

    window = basis.dropna(how="any")
    if window.empty or window.shape[1] == 0:
        raise ValueError("Insufficient data to estimate static RP weights (risk_basis).")

    if diag_only:
        w_hat = _inv_vol_weights(window, long_only=long_only).values
    else:
        S = window.cov().values + ridge * np.eye(window.shape[1])
        w0 = _inv_vol_weights(window, long_only=long_only).values
        w_hat, ok = _erc_weights(S, w0, long_only=long_only)
        if not ok:
            w_hat = _inv_vol_weights(window, long_only=long_only).values

    snap = pd.DataFrame([w_hat], columns=window.columns, index=[pnl.index[0]]).reindex(columns=pnl.columns).fillna(0.0)
    W = _forward_fill_weights(snap, pnl.index).fillna(0.0)
    return _portfolio_from_weights(pnl, W)

def risk_parity_dynamic(
    pnl: pd.DataFrame,
    *,
    risk_basis: Optional[pd.DataFrame] = None,  # strategy series to estimate rolling risk
    lookback: int = 252,
    freq: RebalFreq = "Q",
    long_only: bool = True,
    ridge: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-8,
    diag_only: bool = False,
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    DYNAMIC Risk Parity (no look-ahead):
      - At rebalance date d, uses risk_basis data up to d_prev.
      - Weights applied to 'pnl'.
    """
    pnl = pnl.copy().sort_index()
    basis = (risk_basis if risk_basis is not None else pnl)
    pnl, basis, common_cols = _align_two(pnl, basis)

    cols = pnl.columns
    rb = _rebal_dates(pnl.index, freq=freq)
    w_snaps = []

    for d in rb:
        pos = pnl.index.get_loc(d)
        if pos < 1 or pos < lookback:
            continue
        d_prev = pnl.index[pos - 1]

        # trailing window on the RISK BASIS (exclude d to avoid look-ahead)
        b_win = basis.loc[basis.index[max(0, pos - lookback) : pos - 1]]
        b_win = b_win.dropna(axis=1, how="all")
        if b_win.shape[1] == 0:
            continue

        live = b_win.columns

        if diag_only:
            w = _inv_vol_weights(b_win, long_only=long_only).reindex(live).values
        else:
            S = b_win.cov().values + ridge * np.eye(b_win.shape[1])
            w0 = _inv_vol_weights(b_win, long_only=long_only).reindex(live).values
            w, ok = _erc_weights(S, w0, long_only=long_only, max_iter=max_iter, tol=tol)
            if not ok:
                w = _inv_vol_weights(b_win, long_only=long_only).reindex(live).values

        w_snaps.append(pd.DataFrame(w, index=live, columns=[d]).T.reindex(columns=cols))

    if not w_snaps:
        raise ValueError("No rebalance dates satisfied warmup; adjust lookback/freq or coverage.")
    W = pd.concat(w_snaps).fillna(0.0)
    W = _forward_fill_weights(W, pnl.index).fillna(0.0)
    return _portfolio_from_weights(pnl, W)

# =====================================================================
# 3) Mean–Variance Optimization (strategy-specific risk; optional μ basis)
# =====================================================================

def mvo(
    pnl: pd.DataFrame,
    *,
    risk_basis: Optional[pd.DataFrame] = None,  # covariance source (e.g., MA/Carry PnL)
    mu_basis: Optional[pd.DataFrame] = None,    # optional: expected-return source (defaults to pnl)
    lookback: int = 252,
    freq: RebalFreq = "M",
    long_only: bool = True,
    cap: Optional[float] = None,
    gamma: float = 5.0,
    ridge: float = 1e-6,
    iters: int = 500,
    step: float = 0.1,
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Rolling mean–variance optimizer:
      - Σ estimated from `risk_basis` (strategy-specific risk).
      - μ from `mu_basis` if provided, else from `pnl`.
      - No look-ahead: at date d use data up to d-1.
    """
    pnl = pnl.copy().sort_index()
    rbasis = (risk_basis if risk_basis is not None else pnl)
    mbasis = (mu_basis   if mu_basis   is not None else pnl)

    # Align all three: pnl vs rbasis, then vs mbasis
    pnl, rbasis, common_cols = _align_two(pnl, rbasis)
    pnl, mbasis, common_cols = _align_two(pnl, mbasis)
    cols = pnl.columns

    rb = _rebal_dates(pnl.index, freq=freq)
    w_snaps = []

    for d in rb:
        pos = pnl.index.get_loc(d)
        if pos < 1 or pos < lookback:
            continue
        # trailing windows up to d-1
        r_win = rbasis.iloc[max(0, pos - lookback) : pos]
        m_win = mbasis.iloc[max(0, pos - lookback) : pos]

        r_win = r_win.dropna(axis=1, how="all")
        m_win = m_win.dropna(axis=1, how="all")
        live = sorted(set(r_win.columns) & set(m_win.columns))
        if len(live) == 0:
            continue

        r_win = r_win[live]
        m_win = m_win[live]

        mu = m_win.mean().values
        S  = r_win.cov().values + ridge * np.eye(len(live))

        if not long_only and cap is None:
            invS = _safe_inv(S, ridge=ridge)
            ones = np.ones(len(live))
            A = (invS @ mu) / gamma
            B = invS @ ones
            lam = (ones @ A - 1.0) / (ones @ B + 1e-16)
            w = A - lam * B
            s = w.sum()
            w = w / (s + 1e-16)
        else:
            n = len(live)
            w = np.ones(n) / n
            ones = np.ones(n)
            cap_val = cap if cap is not None else 1.0
            for _ in range(iters):
                grad = gamma * (S @ w) - mu
                w = w - step * grad
                w = np.clip(w, 0.0, cap_val)
                s = w.sum()
                w = w / s if s > 0 else ones / n
            w = w / (w.sum() + 1e-16)

        w_snaps.append(pd.DataFrame(w, index=live, columns=[d]).T.reindex(columns=cols))

    if not w_snaps:
        raise ValueError("No valid rebalance snapshots; check lookback/freq or data coverage.")
    W = pd.concat(w_snaps).fillna(0.0)
    W = _forward_fill_weights(W, pnl.index).fillna(0.0)
    return _portfolio_from_weights(pnl, W)
