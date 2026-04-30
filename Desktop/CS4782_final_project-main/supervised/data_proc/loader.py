from torch.utils.data import DataLoader

from .dataset import TimeSeriesDataset


def dloader(train_data, val_data, test_data, batch_size=32, L=336, T=96):

    # 1. Create Dataset instances
    train_dataset = TimeSeriesDataset(train_data.values, seq_len=L, pred_len=T)
    val_dataset = TimeSeriesDataset(val_data.values, seq_len=L, pred_len=T)
    test_dataset = TimeSeriesDataset(test_data.values, seq_len=L, pred_len=T)

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Quick check on the first batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"Input shape (Batch, L, M): {x_batch.shape}")
    print(f"Target shape (Batch, T, M): {y_batch.shape}")

    return train_loader, val_loader, test_loader
