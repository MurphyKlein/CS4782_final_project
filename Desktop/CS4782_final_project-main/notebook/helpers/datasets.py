import torch
from torch.utils.data import DataLoader, Dataset


class WeatherDataset(Dataset):
    def __init__(self, dataframe, seq_len=512, pred_len=96):
        self.data = dataframe.values.astype("float32")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len

    def __len__(self):
        return len(self.data) - self.total_len + 1

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_len]
        y = self.data[index + self.seq_len : index + self.total_len]
        x = torch.tensor(x).transpose(0, 1)
        y = torch.tensor(y).transpose(0, 1)
        return x, y


def make_loaders(train, val, test, seq_len, pred_len, batch_size=32):
    def loader(arr, shuffle):
        ds = WeatherDataset(arr, seq_len, pred_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader(train, True), loader(val, False), loader(test, False)
