import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .dataset import TimeSeriesDataset


def build_loaders(
    data: np.ndarray,
    L: int,
    T: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple:
    """
    Split data into train/val/test, fit a StandardScaler on train only, and
    return (train_loader, val_loader, test_loader, scaler).
    """
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_raw = data[:n_train]
    val_raw = data[n_train : n_train + n_val]
    test_raw = data[n_train + n_val :]

    scaler = StandardScaler().fit(train_raw)
    train_sc = scaler.transform(train_raw)
    val_sc = scaler.transform(val_raw)
    test_sc = scaler.transform(test_raw)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_dl = DataLoader(
        TimeSeriesDataset(train_sc, L, T),
        shuffle=True,
        **loader_kwargs,
    )
    val_dl = DataLoader(
        TimeSeriesDataset(val_sc, L, T),
        shuffle=False,
        **loader_kwargs,
    )
    test_dl = DataLoader(
        TimeSeriesDataset(test_sc, L, T),
        shuffle=False,
        **loader_kwargs,
    )

    return train_dl, val_dl, test_dl, scaler

