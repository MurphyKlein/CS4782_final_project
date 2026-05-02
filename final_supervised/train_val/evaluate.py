import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute MSE and MAE on a DataLoader.

    Returns:
        (mse, mae) as Python floats.
    """
    model.eval()
    all_pred, all_true = [], []

    for x, y in loader:
        pred = model(x.to(device)).cpu()
        all_pred.append(pred)
        all_true.append(y)

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)

    mse = F.mse_loss(preds, trues).item()
    mae = (preds - trues).abs().mean().item()
    return mse, mae

