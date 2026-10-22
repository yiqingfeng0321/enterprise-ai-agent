from pathlib import Path

import pandas as pd


def extract_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw input file does not exist: {path}")
    return pd.read_csv(path)
