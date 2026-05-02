import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_weather(directory: Path) -> np.ndarray:
    """
    Load all weather CSV files from directory into one long time series.

    CSV files are appended row-wise, sorted by their date/datetime column when
    present, deduplicated by timestamp, restricted to numeric columns, and
    returned as a float32 numpy array.
    """
    directory = Path(directory)
    print(f"Searching for CSVs in: {directory}")
    if directory.exists():
        print("Contents of directory:", [path.name for path in directory.iterdir()])
    else:
        print("Directory does not exist!")

    all_files = sorted(glob.glob(str(directory / "*.csv")))
    if not all_files:
        all_files = sorted(glob.glob(str(directory / "*" / "*.csv")))

    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. Please check your Drive path."
        )

    frames = []
    for fp in all_files:
        df = pd.read_csv(fp, encoding="unicode_escape", low_memory=False)
        print(f"  Loaded {Path(fp).name:40s} shape={df.shape}")
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(
                df[date_col],
                format="mixed",
                dayfirst=True,
            )
            df = df.set_index(date_col)
        frames.append(df.select_dtypes(include=[np.number]))

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"\nFinal combined shape: {combined.shape}")
    return combined.values.astype(np.float32)
