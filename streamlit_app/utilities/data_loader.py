import os
import sqlite3
import pandas as pd
import requests

def download_file_from_google_drive(id, destination):
    import gdown
    url = f"https://drive.google.com/uc?id={id}"
    gdown.download(url, destination, quiet=False)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def ensure_database(file_path="commodities.db", file_id=None):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 10_000_000:
        print(f"[DEBUG] DB already exists: {file_path}")
        return file_path

    if file_id is None:
        raise ValueError("Missing Google Drive file ID to download database.")

    print("[DEBUG] Downloading commodities.db from Google Drive...")
    download_file_from_google_drive(file_id, file_path)

    size = os.path.getsize(file_path)
    print(f"[DEBUG] Downloaded file size: {size / 1_000_000:.2f} MB")
    
    if size < 10_000_000:
        with open(file_path, "rb") as f:
            print("[DEBUG] First 100 bytes of file (for diagnostics):")
            print(f.read(100))
        raise ValueError("Downloaded file appears corrupted or incomplete.")

    print("[DEBUG] Download complete.")
    return file_path


# Core data loading functions
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

def load_filtered_commodities(db_path, internal_tag=None, exchange=None, ticker_search=None, only_metadata=False):
    conn = sqlite3.connect(db_path)

    base_query = "SELECT * FROM futures WHERE 1=1"
    meta_query = """
        SELECT bbg_ticker, description
        FROM futures
        WHERE px_last IS NOT NULL
    """
    params = []
    meta_params = []

    if internal_tag:
        base_query += " AND lower(internal_tag) = ?"
        meta_query += " AND lower(internal_tag) = ?"
        params.append(internal_tag.lower())
        meta_params.append(internal_tag.lower())
    if exchange:
        base_query += " AND lower(exchange) = ?"
        meta_query += " AND lower(exchange) = ?"
        params.append(exchange.lower())
        meta_params.append(exchange.lower())
    if ticker_search:
        base_query += " AND lower(bbg_ticker) LIKE ?"
        meta_query += " AND lower(bbg_ticker) LIKE ?"
        like = f"%{ticker_search.lower()}%"
        params.append(like)
        meta_params.append(like)

    if only_metadata:
        query = f"""
            SELECT bbg_ticker, description
            FROM ({meta_query})
            GROUP BY bbg_ticker
            HAVING COUNT(*) > 0
        """
        df = pd.read_sql(query, conn, params=meta_params)
        conn.close()
        return df.dropna().drop_duplicates().reset_index(drop=True)

    df = pd.read_sql(base_query, conn, params=params, parse_dates=["date"])
    conn.close()
    return {"futures": df}

def get_filter_options(db_path):
    conn = sqlite3.connect(db_path)
    internal_tags = pd.read_sql(
        """
        SELECT DISTINCT internal_tag
        FROM futures
        WHERE internal_tag IS NOT NULL
        AND px_last IS NOT NULL
        """, conn
    )["internal_tag"].dropna().unique().tolist()

    exchanges = pd.read_sql(
        """
        SELECT DISTINCT exchange
        FROM futures
        WHERE exchange IS NOT NULL
        AND px_last IS NOT NULL
        """, conn
    )["exchange"].dropna().unique().tolist()

    conn.close()
    return internal_tags, exchanges
