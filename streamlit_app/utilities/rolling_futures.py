import pandas as pd
from pandas.tseries.offsets import BMonthEnd, BDay


def ensure_date_column(df, date_col="date"):
    """Ensures 'date' column is present and in datetime format."""
    if date_col not in df.columns:
        for col in df.columns:
            if "date" in col.lower():
                df = df.rename(columns={col: date_col})
                break
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' column not found in DataFrame.")
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def pivot_long_to_wide(df, date_col="date", tenor_col="tenor", price_col="price"):
    """Converts long-format futures data to wide format (tenors as columns)."""
    df = df.rename(columns={
        date_col: "date",
        tenor_col: "tenor",
        price_col: "price"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "tenor"])
    wide_df = df.pivot(index="date", columns="tenor", values="price").reset_index()
    if wide_df.drop(columns="date").isna().all().all():
        raise ValueError("Pivoted futures table is empty — no price data found.")
    return wide_df


def mark_roll_events(df, roll_date_col="roll_date", date_col="date"):
    """Marks a binary column Roll_Event where date == T-5 Roll Date."""
    df = df.copy()
    df["Roll_Event"] = (df[date_col] == df[roll_date_col]).astype(int)
    return df


def calculate_rolling_futures_indexed(df, f1_col, f2_col, roll_col, transaction_cost=0.002):
    """
    Spreadsheet-accurate rolling logic:
    - If Roll_Event == 1 → return = F2[today] - F1[yesterday]
    - Else              → return = F1[today] - F1[yesterday]
    - If first row      → entry at F1[0] - cost
    """
    df = df.reset_index(drop=True)

    rolling_equity = []
    pnl_series = []
    cumulative_costs = []
    total_cost = 0
    debug_rows = []

    for idx in range(len(df)):
        t_cost = 0

        if idx == 0:
            equity = df.iloc[idx][f1_col] - transaction_cost
            pnl = 0
            total_cost += transaction_cost
        else:
            curr_date = df.iloc[idx]["date"]
            prev_date = df.iloc[idx - 1]["date"]

            if df.iloc[idx][roll_col] == 1:
                f2_today = df.iloc[idx][f2_col]
                f1_prev = df[df["date"] == prev_date][f1_col].values[0]
                delta = f2_today - f1_prev
                t_cost = 2 * transaction_cost
            else:
                f1_today = df.iloc[idx][f1_col]
                f1_prev = df[df["date"] == prev_date][f1_col].values[0]
                delta = f1_today - f1_prev
                t_cost = 0
            pnl = delta - t_cost
            equity = rolling_equity[-1] + pnl
            total_cost += t_cost

        rolling_equity.append(equity)
        pnl_series.append(pnl)
        cumulative_costs.append(total_cost)

        # Save debug snapshot
        debug_rows.append({
            "date": df.iloc[idx]["date"],
            "Roll": df.iloc[idx][roll_col],
            "F1_today": df.iloc[idx][f1_col],
            "F2_today": df.iloc[idx][f2_col],
            "F1_yest": df.iloc[idx - 1][f1_col] if idx > 0 else None,
            "Delta": delta if idx > 0 else None,
            "t_cost": t_cost,
            "Equity": equity
        })

    df["Rolling Futures"] = rolling_equity
    df["Rolling PnL"] = pnl_series
    df["Cumulative Transaction Costs"] = cumulative_costs

    # Export debug frame for validation
    debug_df = pd.DataFrame(debug_rows)
    df["_debug"] = debug_df.to_dict(orient="records")

    return df


def compute_rolling_futures(
    df_long,
    transaction_cost=0.002,
    date_col="date",
    tenor_col="tenor",
    price_col="price",
    f1_col="F1",
    f2_col="F2"
):
    """Main wrapper to compute rolling futures with dynamic format detection."""
    df_long = ensure_date_column(df_long, date_col)
    df_long = df_long.sort_values(by=date_col).reset_index(drop=True)

    # === FORMAT DETECTION ===
    is_long_format = tenor_col in df_long.columns and price_col in df_long.columns

    if is_long_format:
        wide_df = pivot_long_to_wide(df_long, date_col, tenor_col, price_col)
    else:
        wide_df = df_long.copy()

    # === Select Tenor Pair ===
    tenors = [col for col in wide_df.columns if col != "date"]
    if len(tenors) < 2:
        raise ValueError("Need at least two tenors to compute a rolling futures series.")

    found_pair = False
    for i in range(len(tenors) - 1):
        col1, col2 = tenors[i], tenors[i + 1]
        wide_df[col1] = pd.to_numeric(wide_df[col1], errors="coerce")
        wide_df[col2] = pd.to_numeric(wide_df[col2], errors="coerce")
        tmp = wide_df[[col1, col2]].dropna()
        if not tmp.empty:
            wide_df = wide_df.rename(columns={col1: f1_col, col2: f2_col})
            found_pair = True
            break

    if not found_pair:
        raise ValueError("No valid F1-F2 tenor pair with non-NaN values found.")

    # === Drop invalid rows ===
    wide_df = wide_df.dropna(subset=[f1_col, f2_col])

    # === Mark Roll Events ===
    wide_df["eom"] = wide_df["date"] + BMonthEnd(0)
    wide_df["roll_date"] = wide_df["eom"] - BDay(5)
    wide_df = mark_roll_events(wide_df, roll_date_col="roll_date", date_col="date")

    # === Compute PnL ===
    result_df = calculate_rolling_futures_indexed(
        wide_df, f1_col=f1_col, f2_col=f2_col,
        roll_col="Roll_Event", transaction_cost=transaction_cost
    )

    if result_df["Rolling Futures"].isna().all():
        raise ValueError("All Rolling Futures values are NaN — check tenor overlap or data sparsity.")

    return result_df.dropna(subset=["Rolling Futures"])
