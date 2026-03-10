import torch
import torch.nn as nn
from early_stopping import EarlyStopping
from config import EARLY_STOPPING_PATIENCE

def validate(model, loader, device):

    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_total = 0
    n = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = images.size(0)
            loss_total += loss.item() * batch_size
            n += batch_size

    return loss_total / n


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device, model_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        path=model_path
    )

    logs = {"train_loss": [], "val_loss": []}
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0
        n = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = images.size(0)
            train_loss_total += loss.item() * batch_size
            n += batch_size

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_loss_total / n
        val_loss = validate(model, val_loader, device)

        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.stop:
            break

        scheduler.step()

    return logs