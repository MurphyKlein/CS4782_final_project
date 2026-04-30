from .datasets import make_loaders
from .models import PatchTST_Supervised
from .training import evaluate, freeze_backbone, run_training, unfreeze_all
from .data import DEVICE
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


def run_linear_probing_experiment(
    train_data,
    val_data,
    test_data,
    n_channels,
    seq_len=512,
    pred_len=96,
    patch_len=12,
    stride=12,
    d_model=128,
    n_heads=16,
    n_layers=3,
    d_ff=256,
    dropout=0.2,
    batch_size=32,
    pretrain_epochs=100,
    linprobe_epochs=20,
    finetune_epochs=20,
    pretrain_lr=1e-4,
    linprobe_lr=1e-4,
    finetune_lr=1e-5,
    mask_ratio=0.4,
    dataset_name="dataset",
):
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name} | pred_len={pred_len} | channels={n_channels}")
    print(f"{'=' * 60}")

    tr_loader, val_loader, te_loader = make_loaders(
        train_data,
        val_data,
        test_data,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
    )

    model = PatchTST_Supervised(
        num_features=n_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    print(f"\n--- Step 1: Pretraining ({pretrain_epochs} epochs) ---")
    optimizer_pre = Adam(model.parameters(), lr=pretrain_lr, weight_decay=1e-4)
    scheduler_pre = CosineAnnealingLR(optimizer_pre, T_max=pretrain_epochs)

    model.train()
    for epoch in range(1, pretrain_epochs + 1):
        total_loss = 0
        for x, _ in tr_loader:
            x = x.to(DEVICE)
            optimizer_pre.zero_grad()
            loss = model(x, mode="pretrain")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_pre.step()
            total_loss += loss.item()
        scheduler_pre.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [Pretrain] epoch {epoch:3d}/{pretrain_epochs} | loss={total_loss / len(tr_loader):.4f}")

    print(f"\n--- Step 2: Linear probing ({linprobe_epochs} epochs) ---")
    freeze_backbone(model)
    run_training(
        model,
        tr_loader,
        val_loader,
        n_epochs=linprobe_epochs,
        lr=linprobe_lr,
        mode="forecast",
        desc="LinProbe",
    )

    print(f"\n--- Step 3: Fine-tuning ({finetune_epochs} epochs) ---")
    unfreeze_all(model)
    run_training(
        model,
        tr_loader,
        val_loader,
        n_epochs=finetune_epochs,
        lr=finetune_lr,
        mode="forecast",
        desc="FineTune",
    )

    test_mse, test_mae = evaluate(model, te_loader)
    print(f"\n  ✓  Test  MSE={test_mse:.4f}  MAE={test_mae:.4f}")
    print(
        f"  (Paper Table 4 reference — pred_len={pred_len}: "
        f"Weather≈0.144/0.193, Electricity≈0.126/0.221)"
    )

    return model, test_mse, test_mae
