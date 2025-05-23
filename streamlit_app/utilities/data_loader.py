import os
import sqlite3
import pandas as pd
import requests

# Google Drive download utilities
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={"id": id, "confirm": token}, stream=True)
    save_response_content(response, destination)

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

# Ensure DB file is present and valid
def ensure_database(file_path="commodities.db", file_id=None):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 10_000_000:
        return file_path

    if file_id is None:
        raise ValueError("Missing Google Drive file ID to download database.")

    print("[DEBUG] Downloading commodities.db from Google Drive...")
    download_file_from_google_drive(file_id, file_path)

    if os.path.getsize(file_path) < 10_000_000:
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

def load_filtered_commodities(db_path, internal_tag=None, exchange=None, ticker_search=None):
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
    conn = sqlite3.connect(db_path)
    internal_tags = pd.read_sql(
        "SELECT DISTINCT internal_tag FROM futures WHERE internal_tag IS NOT NULL ORDER BY internal_tag",
        conn)["internal_tag"].dropna().tolist()
    exchanges = pd.read_sql(
        "SELECT DISTINCT exchange FROM futures WHERE exchange IS NOT NULL ORDER BY exchange",
        conn)["exchange"].dropna().tolist()
    conn.close()
    return internal_tags, exchanges
