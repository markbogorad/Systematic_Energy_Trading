import pandas as pd
import re

def drop_dupes(df: pd.DataFrame, n: int = 12, decimals: int = 6) -> pd.DataFrame:
    df = df.sort_index()
    cols = [c for c in df.columns if re.fullmatch(r"F\d+", str(c))]
    front = [c for c in cols if int(c[1:]) <= n]

    keys = df[front].round(decimals).agg(tuple, axis=1)
    dup_mask = keys == keys.shift(1)          # True if same as previous
    keep_mask = ~dup_mask | dup_mask.shift(-1).fillna(False)  # keep later of pair

    cleaned = df.loc[keep_mask].copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    return cleaned