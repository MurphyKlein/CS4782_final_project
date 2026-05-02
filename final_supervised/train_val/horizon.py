import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from final_supervised.models import PatchTST
from final_supervised.utils import load_checkpoint, save_checkpoint

from .evaluate import evaluate
from .train import run_epoch


@dataclass(frozen=True)
class TrainingConfig:
    L: int
    P: int
    S: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    batch_size: int
    lr: float
    max_epochs: int
    patience: int
    lr_patience: int
    lr_min: float
    train_ratio: float
    val_ratio: float
    num_workers: int = 2
    pin_memory: bool = True


def train_one_horizon(
    data: np.ndarray,
    M: int,
    T: int,
    config: TrainingConfig,
    ckpt_dir: Path,
    device: torch.device,
) -> dict:
    """
    Train and evaluate one supervised PatchTST model for a single horizon.
    """
    from final_supervised.data_proc import build_loaders

    print(f'\n{"=" * 64}')
    print(f"  Supervised PatchTST/42   T = {T}")
    print(
        f"  L={config.L}  P={config.P}  S={config.S}  "
        f"N={math.floor((config.L - config.P) / config.S) + 2} patches"
    )
    print(f'{"=" * 64}')

    train_dl, val_dl, test_dl, _ = build_loaders(
        data,
        config.L,
        T,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    print(
        f"  Samples  train={len(train_dl.dataset):,}  "
        f"val={len(val_dl.dataset):,}  "
        f"test={len(test_dl.dataset):,}"
    )

    model = PatchTST(
        M=M,
        L=config.L,
        T=T,
        P=config.P,
        S=config.S,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_patience,
        factor=0.5,
        min_lr=config.lr_min,
    )

    best_val_mse = float("inf")
    patience_count = 0
    best_epoch = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model,
            train_dl,
            device=device,
            optimizer=optimizer,
            is_train=True,
        )
        val_mse, val_mae = evaluate(model, val_dl, device=device)

        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "lr": current_lr,
            }
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_mse, T, ckpt_dir)
        else:
            patience_count += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  ep {epoch:3d}/{config.max_epochs} | "
                f"train={train_loss:.4f} | "
                f"val MSE={val_mse:.4f} MAE={val_mae:.4f} | "
                f"lr={current_lr:.1e} | "
                f"{elapsed:.1f}s | "
                f"patience={patience_count}/{config.patience}"
            )

        if patience_count >= config.patience:
            print(
                f"\n  Early stop triggered at epoch {epoch}."
                f"  Best epoch = {best_epoch}  "
                f"Best val MSE = {best_val_mse:.4f}"
            )
            break

    print(f"\n  Reloading best checkpoint (epoch {best_epoch}) ...")
    load_checkpoint(ckpt_dir / f"patchtst_T{T}_best.pt", model, device=device)
    test_mse, test_mae = evaluate(model, test_dl, device=device)

    print(f"\n  TEST RESULT   MSE = {test_mse:.4f}   MAE = {test_mae:.4f}")

    return {
        "T": T,
        "mse": round(test_mse, 4),
        "mae": round(test_mae, 4),
        "best_epoch": best_epoch,
        "best_val_mse": round(best_val_mse, 4),
        "history": history,
    }
