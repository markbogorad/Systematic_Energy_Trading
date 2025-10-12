from pathlib import Path
import re
import pandas as pd

def read_data(file_path: str | Path, sheet: str) -> pd.DataFrame:
    
    # --- Resolve and read ---
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # --- Locate date column ---
    date_col = next((c for c in df.columns if str(c).strip().lower() in {"date", "dates"}), None)
    if date_col is None:
        raise ValueError("No 'Date' or 'Dates' column found in the sheet.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    
    # --- Normalize column names (e.g. '_1 Comdty' → 'F1') ---
    rename_map = {}
    for c in df.columns:
        s = str(c).strip()
        
        # Case 1: Bloomberg style "BAP1 Comdty"
        if match := re.search(r"(\d+)\s*Comdty$", s, re.IGNORECASE):
            rename_map[c] = f"F{match.group(1)}"
        # Case 2: Already short or clean ("F1", "M1", "1")
        elif match := re.fullmatch(r"[FfMm]?(\d+)", s):
            rename_map[c] = f"F{match.group(1)}"
    
    df = df.rename(columns=rename_map)
    
    # --- Keep and order tenor columns ---
    tenor_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"F\d+", c)],
        key=lambda x: int(x[1:])
    )
    
    if not tenor_cols:
        raise ValueError("No tenor columns detected (expected headers like 'F1', 'F2', ...).")
    
    df = df[tenor_cols]
    
    return df