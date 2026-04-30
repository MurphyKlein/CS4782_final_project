import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import DEVICE, device
from .datasets import make_loaders
from .models import PatchTST_Supervised


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, loader, optimizer, mode="forecast"):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if mode == "forecast":
            pred = model(x, mode="forecast")
            loss = F.mse_loss(pred, y)
        else:
            loss = model(x, mode="pretrain")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mse_total, mae_total, count = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x, mode="forecast")
        mse_total += F.mse_loss(pred, y, reduction="sum").item()
        mae_total += F.l1_loss(pred, y, reduction="sum").item()
        count += y.numel()
    return mse_total / count, mae_total / count


def run_training(model, train_loader, val_loader, n_epochs, lr, mode="forecast", desc=""):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_val, best_state = float("inf"), None

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, mode=mode)
        scheduler.step()

        if mode == "forecast":
            val_mse, val_mae = evaluate(model, val_loader)
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[{desc}] epoch {epoch:3d}/{n_epochs} | "
                    f"train_loss={tr_loss:.4f} | val_MSE={val_mse:.4f} | val_MAE={val_mae:.4f}"
                )
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"[{desc}] epoch {epoch:3d}/{n_epochs} | pretrain_loss={tr_loss:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "forecast_head" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def pretrain_model(model, train_loader, val_loader, epochs=20, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Pre-train]"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed, raw, mask = model(batch_x)
            loss = criterion(reconstructed[mask], raw[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f}")

    return model


def pretrain_model_with_es(model, train_loader, val_loader, epochs=20, lr=1e-4, patience=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Pre-train]"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed, raw, mask = model(batch_x)
            loss = criterion(reconstructed[mask], raw[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstructed, raw, mask = model(batch_x)
                val_loss += criterion(reconstructed[mask], raw[mask]).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return model


def linear_probing_with_es(supervised_model, pretrained_model, train_loader, val_loader, epochs=20, lr=1e-4, patience=5):
    supervised_model.to(device)
    supervised_model.load_pre_trained_backbone(pretrained_model.state_dict(), freeze_backbone=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, supervised_model.parameters()), lr=lr)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        supervised_model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Linear Probing]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = supervised_model(batch_x)
            loss = mse_criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        supervised_model.eval()
        val_mse, val_mae = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = supervised_model(batch_x)
                val_mse += mse_criterion(predictions, batch_y).item()
                val_mae += mae_criterion(predictions, batch_y).item()

        avg_val_mse = val_mse / len(val_loader)
        print(
            f"Epoch {epoch + 1} | Train MSE: {train_loss / len(train_loader):.4f} | "
            f"Val MSE: {avg_val_mse:.4f} | Val MAE: {val_mae / len(val_loader):.4f}"
        )

        early_stopping(avg_val_mse)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return supervised_model
