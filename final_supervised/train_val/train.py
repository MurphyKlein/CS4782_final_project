import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    is_train: bool = True,
) -> float:
    """
    Run one full pass through loader and return average MSE over all samples.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)
