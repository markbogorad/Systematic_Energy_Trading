import sqlite3
import pandas as pd

def load_commodity_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    commodity_data = {}
    for (table_name,) in tables:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["Date"])
            df.set_index("date", inplace=True)
            commodity_data[table_name] = df
        except Exception as e:
            print(f"Failed to load table '{table_name}': {e}")
    conn.close()
    return commodity_data

def load_filtered_commodities(db_path, internal_tag=None, exchange=None, ticker_search=None):
    import sqlite3

    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM futures WHERE 1=1"
    params = []

    if internal_tag:
        query += " AND lower(internal_tag) = ?"
        params.append(internal_tag.lower())

    if exchange:
        query += " AND lower(exchange) = ?"
        params.append(exchange.lower())

    if ticker_search:
        query += " AND lower(bbg_ticker) LIKE ?"
        params.append(f"%{ticker_search.lower()}%")

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    return {"futures": df}


def get_filter_options(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)

    internal_tags = pd.read_sql("SELECT DISTINCT internal_tag FROM futures WHERE internal_tag IS NOT NULL ORDER BY internal_tag", conn)["internal_tag"].dropna().tolist()
    exchanges = pd.read_sql("SELECT DISTINCT exchange FROM futures WHERE exchange IS NOT NULL ORDER BY exchange", conn)["exchange"].dropna().tolist()
    
    conn.close()
    return internal_tags, exchanges
