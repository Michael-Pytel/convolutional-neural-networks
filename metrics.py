import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_predictions(model, loader, device):

    model.eval()

    preds = []
    labels_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            labels_all.extend(labels.numpy())

    cm = confusion_matrix(labels_all, preds)

    return cm


def confusion_stats(cm):

    TP = np.diag(cm)

    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    return {
        "TP": TP.tolist(),
        "FP": FP.tolist(),
        "FN": FN.tolist(),
        "TN": TN.tolist()
    }