import os
import sqlite3
import pandas as pd
import gdown

# === Lowercase DB key map ===
#TAG_DB_MAP = {

    #"basics":        ("commodities_basics.db", "1CmUdyLvKneqLIzFnBO-c20d7oOaZvlzY"),
    #"catherine rec":    ("commodities_cme_liquid.db", "1fyeaWrRC0tnb2kM-Vglq9a8ai-vNv-7p"),
    #"cme liquid": ("commodities_catherine_rec.db", "1TbOf3L6gVzL0QWon_XHoKZmOx9CK2nYT"),
    #"liquid ice us": ("commodities_liquid_ice_us.db", "18vg6zzASt2xXefmOpAZU20UC9XjhiWZl"),
#}

# === CONFIG ===
DB_FILENAME = "NGLs.db"
DB_FILE_ID = "1riBajOwyzgODJBGDMaeLV9BaxaIQDk_M"
TMP_DIR = "/tmp"

def ensure_ngls_database(tmp_dir=TMP_DIR):
    local_path = os.path.join(tmp_dir, DB_FILENAME)
    if os.path.exists(local_path):
        return local_path

    print(f"[DEBUG] Downloading {DB_FILENAME} from Google Drive...")
    url = f"https://drive.google.com/uc?id={DB_FILE_ID}"
    gdown.download(url, local_path, quiet=False)
    return local_path

def load_commodity_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    commodity_data = {}
    for (table_name,) in tables:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["date"])
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            commodity_data[table_name] = df
        except Exception as e:
            print(f"Failed to load table '{table_name}': {e}")
    conn.close()
    return commodity_data

def load_filtered_commodities(db_path, ticker_search=None, only_metadata=False):
    conn = sqlite3.connect(db_path)

    if only_metadata:
        query = """
            SELECT m.bbg_ticker, m.name as description
            FROM metadata m
            WHERE EXISTS (
                SELECT 1 FROM futures f
                WHERE f.bbg_ticker = m.bbg_ticker
                AND f.px_last IS NOT NULL
                AND f.date IS NOT NULL
                AND f.tenor IS NOT NULL
            )
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df.dropna().drop_duplicates().reset_index(drop=True)

    # Get futures data filtered by ticker
    if ticker_search:
        query = """
            SELECT *
            FROM futures
            WHERE UPPER(bbg_ticker) = UPPER(?)
            AND px_last IS NOT NULL
        """
        df = pd.read_sql(query, conn, params=[ticker_search], parse_dates=["date"])
    else:
        df = pd.read_sql("SELECT * FROM futures", conn, parse_dates=["date"])

    conn.close()

    if df.empty:
        return None

    return {"futures": df}

def get_filter_options(db_path):
    conn = sqlite3.connect(db_path)

    internal_tags = pd.read_sql(
        """
        SELECT DISTINCT internal_tag
        FROM futures
        WHERE internal_tag IS NOT NULL AND px_last IS NOT NULL
        """, conn
    )["internal_tag"].dropna().unique().tolist()

    exchanges = pd.read_sql(
        """
        SELECT DISTINCT exchange
        FROM futures
        WHERE exchange IS NOT NULL AND px_last IS NOT NULL
        """, conn
    )["exchange"].dropna().unique().tolist()

    conn.close()
    return internal_tags, exchanges
