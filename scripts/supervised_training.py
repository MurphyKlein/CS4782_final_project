import time

import torch
from torch import nn
from torch import optim

from supervised.train_val.early_stopping import EarlyStopping
from supervised.train_val.train_function import train_one_epoch
from supervised.train_val.val_function import evaluate


def sup_train(
    model,
    num_epochs,
    train_loader,
    val_loader,
    test_loader,
    device,
    T,
    model_path,
    targets,
):
    device = torch.device(device)
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    print(f"Starting test run for T={T} on {device}...")
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        # Validation phase
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Epoch: {epoch + 1:02} | "
            f"Time: {epoch_mins:.0f}m {epoch_secs:.0f}s"
        )
        print(f"\tTrain Loss (MSE): {train_loss:.4f}")
        print(
            f"\t Val. Loss (MSE): {val_loss:.4f} | "
            f"Val. MAE: {val_mae:.4f}"
        )

        # Check for early stopping and save best model
        early_stopping(val_loss, model, model_path)

        if early_stopping.early_stop:
            print(f"Early stopping triggered for T={T}. Training halted.")
            break

    # 4. Final Evaluation for this Horizon
    print(f"\nEvaluating final performance for T={T}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    test_mse, test_mae = evaluate(model, test_loader, criterion, device)

    print(f"--- Final Results for T={T} (Replication vs. Paper) ---")
    print(
        f"Replicated MSE: {test_mse:.4f} "
        f"(Paper Target: {targets[T]['mse']})"
    )  # [cite: 217]
    print(
        f"Replicated MAE: {test_mae:.4f} "
        f"(Paper Target: {targets[T]['mae']})"
    )  # [cite: 217]
