import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE


def load_weather(filepath):
    df = pd.read_csv(filepath, encoding="unicode_escape")
    print(f"[Weather] Raw shape: {df.shape}")

    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        df = df.drop(columns=date_cols)

    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    print(f"[Weather] Final shape: {df.shape}")
    return df


def load_electricity(filepath):
    print("[Electricity] Loading... (this may take a moment)")
    df = pd.read_csv(filepath, sep=";", decimal=",", index_col=0, parse_dates=True)
    print(f"[Electricity] Raw shape: {df.shape}")

    df = df.resample("1H").mean()

    cols_with_zeros = df.columns[(df == 0).any()]
    print(f"[Electricity] Dropping {len(cols_with_zeros)} columns containing zeros.")
    df = df.drop(columns=cols_with_zeros)

    df = df.dropna()
    print(f"[Electricity] Final shape: {df.shape}")
    return df


def split_and_scale(df, train_ratio=0.7, val_ratio=0.1):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_train].values.astype(np.float32)
    val = df.iloc[n_train : n_train + n_val].values.astype(np.float32)
    test = df.iloc[n_train + n_val :].values.astype(np.float32)

    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(val), scaler.transform(test), scaler
