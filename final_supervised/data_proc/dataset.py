import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for multivariate time series.

    Returns:
        x: FloatTensor with shape (M, L), the channel-first look-back window.
        y: FloatTensor with shape (M, T), the channel-first forecast target.
    """

    def __init__(self, data: np.ndarray, L: int, T: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.L = L
        self.T = T

    def __len__(self) -> int:
        return len(self.data) - self.L - self.T + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.L]
        y = self.data[idx + self.L : idx + self.L + self.T]
        return x.T, y.T

