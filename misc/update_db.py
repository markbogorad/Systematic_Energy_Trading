import pandas as pd
import sqlite3
import os

DB_PATH = "../Data/commodities.db"
EXCEL_FILE = "../Data/May15.xlsx"  # Single Excel file

def create_tables(conn):
    # Futures price curves
    conn.execute('''
    CREATE TABLE IF NOT EXISTS futures (
        date TEXT,
        tenor TEXT,
        PX_LAST REAL,
        BBG_ticker TEXT,
        commodity TEXT,        -- Optional description (can be NULL)
        family TEXT,           -- Reserved for future classification
        description TEXT       -- Reserved for additional detail
    )
    ''')
    conn.execute('''
    CREATE INDEX IF NOT EXISTS idx_bbg_tenor_date 
    ON futures (BBG_ticker, tenor, date)
    ''')

    # COT data 
    conn.execute('''
    CREATE TABLE IF NOT EXISTS COT_data (
        date TEXT,
        BBG_ticker TEXT,
        net_long REAL,
        open_interest REAL,
        trader_category TEXT
        -- Add more fields as needed later
    )
    ''')
    conn.commit()

def process_excel_file(filepath):
    xls = pd.ExcelFile(filepath)
    all_data = []

    for sheet_name in xls.sheet_names:
        print(f"→ Parsing sheet: {sheet_name}")
        try:
            df = xls.parse(sheet_name, skiprows=3)

            # Skip if sheet is empty or too few columns
            if df.empty or len(df.columns) < 2:
                print(f"Skipped: empty or insufficient columns - {sheet_name}")
                continue

            # Check if first column looks like dates
            sample_first_col = df.iloc[:, 0].dropna().astype(str).head(5)
            if sample_first_col.str.contains('[a-zA-Z]', regex=True).any():
                print(f"Skipped: non-date values in first column - {sheet_name}")
                continue

            # Rename first column to 'date'
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

            # Drop non-date rows
            df = df[pd.to_datetime(df['date'], errors='coerce').notnull()].copy()
            df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

            # Extract tenor columns
            tenor_labels = [f"{i+1}M" for i in range(24)]
            tenor_columns = df.columns[1:25]
            if len(tenor_columns) < 1:
                print(f"Skipped: no tenor columns found - {sheet_name}")
                continue

            tenor_map = dict(zip(tenor_columns, tenor_labels))
            df.rename(columns=tenor_map, inplace=True)

            # Melt
            df_melt = df.melt(id_vars=["date"], var_name="tenor", value_name="PX_LAST")
            df_melt["BBG_ticker"] = sheet_name
            df_melt["commodity"] = None
            df_melt["family"] = None
            df_melt["description"] = None

            all_data.append(df_melt)

        except Exception as e:
            print(f"Error processing sheet '{sheet_name}': {e}")
            continue

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)


def insert_futures_data(df, conn):
    df.to_sql("futures", conn, if_exists="append", index=False)
    print(f"Inserted {len(df)} rows into 'futures'.")

def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    print(f"Processing: {EXCEL_FILE}")
    df = process_excel_file(EXCEL_FILE)
    if df is None or df.empty:  
        print("No valid data found to insert.")
    else:
        insert_futures_data(df, conn)
    conn.close()

if __name__ == "__main__":
    main()
