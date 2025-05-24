import os
import sqlite3
import pandas as pd
import gdown

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

# === Tag to DB Mapping ===
TAG_DB_MAP = {
    "Basics":        ("commodities_basics.db", "1CmUdyLvKneqLIzFnBO-c20d7oOaZvlzY"),
    "CME Liquid":    ("commodities_cme_liquid.db", "1fyeaWrRC0tnb2kM-Vglq9a8ai-vNv-7p"),
    "Catherine Rec": ("commodities_catherine_rec.db", "1TbOf3L6gVzL0QWon_XHoKZmOx9CK2nYT"),
    "Liquid ICE US": ("commodities_liquid_ice_us.db", "18vg6zzASt2xXefmOpAZU20UC9XjhiWZl"),
}

def ensure_database_by_tag(internal_tag, tmp_dir="/tmp"):
    if internal_tag not in TAG_DB_MAP:
        raise ValueError(f"No database found for internal_tag: {internal_tag}")

    filename, file_id = TAG_DB_MAP[internal_tag]
    local_path = os.path.join(tmp_dir, filename)

    if os.path.exists(local_path):
        return local_path

    print(f"[DEBUG] Downloading {filename} for tag: {internal_tag}")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, local_path, quiet=False)
    return local_path


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
