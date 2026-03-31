import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_predictions(model, loader, device):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            labels_all.extend(labels.numpy())

    return confusion_matrix(labels_all, preds)


def confusion_stats(cm):
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = TP.sum() / cm.sum()

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),

        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }