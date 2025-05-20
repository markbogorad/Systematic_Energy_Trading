import pandas as pd
from pandas.tseries.offsets import MonthEnd, BDay

# === 0. Safe Date Column Detection ===
def ensure_date_column(df, date_col="Date"):
    if date_col not in df.columns:
        for col in df.columns:
            if "date" in col.lower():
                df = df.rename(columns={col: date_col})
                break
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' column not found in DataFrame.")
    df[date_col] = pd.to_datetime(df[date_col])
    return df

# === 1. End-of-Month (EOM) Rolling Calendar ===
def define_eom_roll_calendar(df, date_col="Date"):
    df = ensure_date_column(df.copy(), date_col)
    df['Roll'] = (df[date_col] == (df[date_col] + MonthEnd(0))).astype(int)
    df['Action'] = 'Hold'
    df.loc[df.index[0], 'Action'] = 'Buy'
    df.loc[df['Roll'] == 1, 'Action'] = 'Sell & Buy'
    df.loc[df.index[-1], 'Action'] = 'Sell'
    df['Holding'] = 'F1'
    df.loc[df['Roll'] == 1, 'Holding'] = 'F2'
    return df

# === 2. Rolling Calendar Based on Expiry File ===
def load_calendar_and_define_roll(filepath, df, date_col="Date", t_minus_days=5):
    df = ensure_date_column(df.copy(), date_col)
    calendar = pd.read_excel(filepath, sheet_name="Expiration Calendar", header=None)
    calendar.columns = ['Expiry']
    calendar['Expiry'] = pd.to_datetime(calendar['Expiry'])
    calendar['Roll_Date'] = calendar['Expiry'] - BDay(t_minus_days)

    adjusted = []
    for roll_date in calendar['Roll_Date']:
        valid = df[df[date_col] <= roll_date][date_col]
        adjusted.append(valid.iloc[-1] if not valid.empty else roll_date)

    calendar['Adjusted_Roll'] = adjusted
    return calendar

def integrate_calendar_roll(df, calendar_df, date_col="Date"):
    df = ensure_date_column(df.copy(), date_col)
    roll_dates = calendar_df['Adjusted_Roll'].tolist()
    expiry_dates = calendar_df['Expiry'].tolist()

    df['Roll'] = 0
    df.loc[df[date_col].isin(roll_dates), 'Roll'] = 1
    df.loc[df[date_col].isin(expiry_dates), 'Roll'] = 2

    holding = []
    current = 'F1'
    for _, row in df.iterrows():
        if row['Roll'] == 1:
            current = 'F2'
        elif row['Roll'] == 2:
            current = 'F1'
        holding.append(current)
    df['Holding'] = holding

    df['Action'] = 'Hold'
    df.loc[df.index[0], 'Action'] = 'Buy'
    df.loc[df['Roll'] == 1, 'Action'] = 'Sell & Buy'
    df.loc[df.index[-1], 'Action'] = 'Sell'
    return df

# === 3. PnL Computation ===
def calculate_rolling_futures(df, transaction_cost=0.002, f1_col="F1", f2_col="F2"):
    df = df.copy()
    pnl, val, cost = [], [], []
    total_cost = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            price = row[f1_col]
            tc = transaction_cost
            adjusted = price - tc
            pnl.append(0)
        else:
            prev = df.iloc[i - 1]
            if row['Roll'] == 1:
                delta = row[f1_col] - prev[f1_col]
                tc = 2 * transaction_cost
            elif row['Roll'] == 2:
                delta = row[f1_col] - prev[f2_col]
                tc = 0
            else:
                if prev['Holding'] == 'F1':
                    delta = row[f1_col] - prev[f1_col]
                else:
                    delta = row[f2_col] - prev[f2_col]
                tc = 0
            adjusted += delta - tc
            pnl.append(delta - tc)

        total_cost += tc
        val.append(adjusted)
        cost.append(total_cost)

    df['Rolling Futures'] = val
    df['Rolling PnL'] = pnl
    df['Cumulative Cost'] = cost
    df['Transaction Cost'] = df['Action'].map({
        'Buy': transaction_cost,
        'Sell': transaction_cost,
        'Sell & Buy': 2 * transaction_cost
    }).fillna(0)
    return df

# === 4. Master Wrapper ===
def compute_rolling_futures(df, method="eom", transaction_cost=0.002, date_col="Date", calendar_path=None, f1_col="F1", f2_col="F2"):
    method = method.lower()
    df = ensure_date_column(df.copy(), date_col)

    if method == "eom":
        df = define_eom_roll_calendar(df, date_col=date_col)
    elif method == "roll mandate + calendar":
        if calendar_path is None:
            raise ValueError("Calendar file path must be provided for calendar-based rolling.")
        calendar_df = load_calendar_and_define_roll(calendar_path, df, date_col=date_col)
        df = integrate_calendar_roll(df, calendar_df, date_col=date_col)
    else:
        raise ValueError("Invalid roll method: choose 'eom' or 'roll mandate + calendar'.")

    return calculate_rolling_futures(df, transaction_cost=transaction_cost, f1_col=f1_col, f2_col=f2_col)
