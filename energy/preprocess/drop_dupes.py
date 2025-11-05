import pandas as pd
import re

def drop_dupes(df: pd.DataFrame, n: int = 12, decimals: int = 6) -> pd.DataFrame:
    """
    Drop consecutive duplicate Bloomberg rows (pairs, triples, longer).
    Rule (current): for any consecutive run of identical F1..Fn values, keep ONLY the FIRST row.
    """
    df = df.sort_index()

    # tenor-like cols: F1, F2, ...
    cols = [c for c in df.columns if re.fullmatch(r"F\d+", str(c))]
    front = [c for c in cols if int(c[1:]) <= n]

    # signature of the front part
    sig = df[front].round(decimals).agg(tuple, axis=1)

    # keep FIRST of each consecutive block
    keep_mask = sig.ne(sig.shift(1))  # True if this row differs from the previous

    cleaned = df.loc[keep_mask].copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    return cleaned
