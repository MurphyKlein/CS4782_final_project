import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        Args:
            data: Numpy array of shape
                (num_timesteps, num_features)
            seq_len: The look-back window (L)
            pred_len: The forecasting horizon (T)
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # We must stop before the end so that we have enough room for the
        # prediction window T.
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # Extract the look-back window L
        s_begin = index
        s_end = s_begin + self.seq_len

        # Extract the prediction window T
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Slice the data: (L, M) and (T, M) where M is the number of features
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
        )
