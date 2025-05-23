import pandas as pd
from pandas.tseries.offsets import MonthEnd


def ensure_date_column(df, date_col="date"):
    """Ensures there is a properly named and typed 'date' column."""
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
    """Converts long-format SQL table to wide format for rolling logic."""
    df = df.rename(columns={
        date_col: "date",
        tenor_col: "tenor",
        price_col: "price"
    })
    df["date"] = pd.to_datetime(df["date"])

    # Drop duplicates if necessary
    df = df.drop_duplicates(subset=["date", "tenor"])

    wide_df = df.pivot(index="date", columns="tenor", values="price").reset_index()

    # Ensure the pivot didn't result in an empty DataFrame
    if wide_df.drop(columns="date").isna().all().all():
        raise ValueError("Pivoted futures table is empty — no price data found.")
    
    return wide_df


def define_eom_roll_calendar(df, date_col="date"):
    """Adds EOM-based rolling logic to the price DataFrame."""
    df = ensure_date_column(df.copy(), date_col)
    df["Roll"] = (df[date_col] == (df[date_col] + MonthEnd(0))).astype(int)
    df["Action"] = "Hold"
    if not df.empty:
        df.loc[df.index[0], "Action"] = "Buy"
        df.loc[df["Roll"] == 1, "Action"] = "Sell & Buy"
        df.loc[df.index[-1], "Action"] = "Sell"
        df["Holding"] = "F1"
        df.loc[df["Roll"] == 1, "Holding"] = "F2"
    return df


def calculate_rolling_futures(df, transaction_cost, f1_col="F1", f2_col="F2"):
    """Simulates rolling futures performance based on defined action and holding columns."""
    if df.empty:
        raise ValueError("No data available for rolling computation.")

    df = df.copy()
    pnl, val, cost = [], [], []
    total_cost = 0

    adjusted = df.iloc[0][f1_col] - transaction_cost
    pnl.append(0)
    val.append(adjusted)
    cost.append(transaction_cost)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if curr["Roll"] == 1:
            delta = curr[f1_col] - prev[f1_col]
            tc = 2 * transaction_cost
        elif curr["Roll"] == 2:
            delta = curr[f1_col] - prev[f2_col]
            tc = 0
        else:
            if prev["Holding"] == "F1":
                delta = curr[f1_col] - prev[f1_col]
            else:
                delta = curr[f2_col] - prev[f2_col]
            tc = 0

        adjusted += delta - tc
        total_cost += tc

        pnl.append(delta - tc)
        val.append(adjusted)
        cost.append(total_cost)

    df["Rolling Futures"] = val
    df["Rolling PnL"] = pnl
    df["Cumulative Cost"] = cost
    df["Transaction Cost"] = df["Action"].map({
        "Buy": transaction_cost,
        "Sell": transaction_cost,
        "Sell & Buy": 2 * transaction_cost
    }).fillna(0)

    return df

def compute_rolling_futures(
    df_long,
    transaction_cost,
    method="eom",
    f1_col="F1",
    f2_col="F2",
    date_col="date",
    tenor_col="tenor",
    price_col="price"
):
    """Wrapper to compute rolling futures from long-format SQL data."""
    if method != "eom":
        raise ValueError("Only 'eom' (End-of-Month) roll method is supported.")

    wide_df = pivot_long_to_wide(df_long, date_col, tenor_col, price_col)

    tenor_cols = [col for col in wide_df.columns if col != "date"]
    found_pair = False

    for i in range(len(tenor_cols) - 1):
        col1, col2 = tenor_cols[i], tenor_cols[i + 1]

        # Convert candidate columns to numeric
        wide_df[col1] = pd.to_numeric(wide_df[col1], errors="coerce")
        wide_df[col2] = pd.to_numeric(wide_df[col2], errors="coerce")

        tmp = wide_df[[col1, col2]].dropna()
        if not tmp.empty:
            wide_df = wide_df.rename(columns={col1: f1_col, col2: f2_col})
            found_pair = True
            break

    if not found_pair:
        raise ValueError("No tenor pair with overlapping non-NaN data found.")

    # Trim front: start from first date where both F1 and F2 exist
    valid_idx = wide_df[[f1_col, f2_col]].dropna().index
    if valid_idx.empty:
        raise ValueError("No usable data found with non-NaN F1 and F2.")
    wide_df = wide_df.loc[valid_idx[0]:].copy()

    wide_df = define_eom_roll_calendar(wide_df, date_col="date")
    result_df = calculate_rolling_futures(wide_df, transaction_cost, f1_col, f2_col)

    return result_df
