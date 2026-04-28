def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        if i == 0:
            print("batch_x device:", batch_x.device)
            print("batch_y device:", batch_y.device)
            print("model device:", next(model.parameters()).device)
            print("patch_embedding device:", model.patch_embedding.weight.device)
            print("pos_embedding device:", model.pos_embedding.device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)