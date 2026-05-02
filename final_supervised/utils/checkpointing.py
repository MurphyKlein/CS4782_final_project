from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mse: float,
    horizon: int,
    ckpt_dir: Path,
    tag: str = "best",
) -> Path:
    """
    Save one training checkpoint to ckpt_dir.

    The notebook calls this only when validation MSE improves, so the default
    tag writes the best checkpoint for each horizon:
    patchtst_T{horizon}_best.pt

    Saved fields:
    {
        "epoch": int,
        "val_mse": float,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"patchtst_T{horizon}_{tag}.pt"
    torch.save(
        {
            "epoch": epoch,
            "val_mse": val_mse,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a checkpoint saved by save_checkpoint.

    The model state is always restored. If an optimizer is provided, its state
    is restored too. The checkpoint is loaded onto device when provided, or CPU
    otherwise.
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(
        f"Loaded checkpoint from {path}  "
        f"(epoch={ckpt['epoch']}, val_mse={ckpt['val_mse']:.4f})"
    )
    return ckpt
