import torch


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            total_mse += criterion(outputs, batch_y).item()
            total_mae += torch.abs(outputs - batch_y).mean().item()

    return total_mse / len(dataloader), total_mae / len(dataloader)
