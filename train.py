import torch
import torch.nn as nn
import numpy as np
from early_stopping import EarlyStopping
from config import EARLY_STOPPING_PATIENCE

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def validate(model, loader, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()

    loss_total, correct, total = 0, 0, 0

    use_amp = device.type == "cuda"
    device_type = "cuda" if use_amp else "cpu"

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss_total += loss.item() * images.size(0)

    return loss_total / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device, model_path,
                mix_type=None, mixup_alpha=1.0, cutmix_alpha=1.0, mix_prob=1, p_mixup=0.5):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, path=model_path)

    logs = {"train_loss": [], "val_loss": [], "val_acc": []}

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss, total = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            if mix_type is not None:
                use_mix = np.random.rand() < mix_prob
            else:
                use_mix = False
            if use_mix:
                if mix_type == "mixup":
                    images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)

                elif mix_type == "cutmix":
                    images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)

                elif mix_type == "both":
                    if np.random.rand() < p_mixup:
                        images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                    else:
                        images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                y_a, y_b, lam = labels, labels, 1



            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = mix_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = total_loss / total
        val_loss, val_acc = validate(model, val_loader, device)

        scheduler.step()

        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.stop:
            break

    return logs