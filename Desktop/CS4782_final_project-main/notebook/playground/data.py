import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y


def make_loaders(train, val, test, seq_len, pred_len, batch_size=32):
    def loader(arr, shuffle):
        ds = TimeSeriesDataset(arr, seq_len, pred_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    return loader(train, True), loader(val, False), loader(test, False)
