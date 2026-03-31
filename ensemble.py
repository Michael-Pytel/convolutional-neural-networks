import torch
import numpy as np
import json
from sklearn.metrics import confusion_matrix


# =========================
# Utilities
# =========================

def normalize_weights(weights):
    w = torch.tensor(weights, dtype=torch.float32)
    return w / (w.sum() + 1e-8)


def get_labels(loader):
    labels = []
    for _, y in loader:
        labels.extend(y.numpy())
    return np.array(labels)


# =========================
# Collect outputs
# =========================

def collect_outputs(models, loader, device):
    all_probs = []

    for model in models:
        model.eval()
        probs = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                out = torch.softmax(model(x), dim=1)
                probs.append(out.cpu())

        all_probs.append(torch.cat(probs))

    return torch.stack(all_probs)  # (n_models, N, C)


# =========================
# Voting
# =========================
def hard_voting(probs):
    """
    probs: (n_models, N, C)
    """
    preds = probs.argmax(dim=2)
    return preds.mode(dim=0).values

def soft_voting(probs):
    return probs.mean(dim=0).argmax(dim=1)


def weighted_voting(probs, weights):
    w = normalize_weights(weights).view(-1, 1, 1)
    return (probs * w).sum(dim=0).argmax(dim=1)


# =========================
# Evaluation
# =========================

def evaluate_preds(preds, labels):
    from metrics import confusion_stats
    cm = confusion_matrix(labels, preds)
    return confusion_stats(cm)


def evaluate_ensemble(models, loader, device, method="soft", weights=None):
    probs = collect_outputs(models, loader, device)

    if method == "soft":
        preds = soft_voting(probs)

    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights required for weighted voting")
        preds = weighted_voting(probs, weights)

    elif method == "hard":
        preds = hard_voting(probs)

    else:
        raise ValueError(f"Unknown method: {method}")

    labels = get_labels(loader)
    return evaluate_preds(preds.numpy(), labels)


# =========================
# Stacking (multi-model)
# =========================

def build_features(probs):
    return probs.permute(1, 0, 2).reshape(probs.shape[1], -1)


def train_stacking(models, val_loader, device, model_type="logreg"):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    probs = collect_outputs(models, val_loader, device)

    X = build_features(probs)
    y = get_labels(val_loader)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=42)

    elif model_type == "ridge":
        model = RidgeClassifier(random_state=42)

    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    elif model_type == "gb":
        model = GradientBoostingClassifier(random_state=42)

    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42)

    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type in ["logreg", "ridge", "svm", "knn"]:
        model.fit(X_scaled, y)
        return model, scaler
    else:
        model.fit(X, y)
        return model, None


def stacking_predict(models, meta_model, scaler, loader, device):
    probs = collect_outputs(models, loader, device)

    X = build_features(probs)

    if scaler is not None:
        X = scaler.transform(X)

    return meta_model.predict(X)


def evaluate_stacking(models, val_loader, test_loader, device, model_type="logreg"):
    meta_model, scaler = train_stacking(models, val_loader, device, model_type)

    preds = stacking_predict(models, meta_model, scaler, test_loader, device)
    labels = get_labels(test_loader)

    return evaluate_preds(preds, labels)


# =========================
# Saving
# =========================

def save_report(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {path}")