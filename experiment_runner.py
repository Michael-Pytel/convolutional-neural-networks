import os
import json
import random
import torch
import numpy as np

from config import RESULTS_DIR, SEEDS, EPOCHS, DEVICE
from utils import set_seed
from datasets import get_dataloaders
from models import get_model
from train import train_model
from metrics import evaluate_predictions, confusion_stats


device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


def sample_param(rng, values):
    if isinstance(values, tuple):
        low, high, scale = values

        if scale == "log":
            return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        elif scale == "linear":
            return float(rng.uniform(low, high))
        else:
            raise ValueError(f"Unknown scale: {scale}")

    val = rng.choice(values)
    if isinstance(val, (np.integer, np.floating)):
        return val.item()

    return val


def random_search(space, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    keys = [k for k in space if k != "model"]

    for _ in range(n_samples):
        yield {
            k: sample_param(rng, space[k])
            for k in keys
        }

def run_single(config, seed, model_path):
    print(f"Seed: {seed}, Config: {json.dumps(config)}")
    set_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        config["batch_size"],
        use_augmentation=config.get("augmentation", True),
        model_name=config["model"]
    )

    model = get_model(config["model"], config["dropout"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logs = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        EPOCHS,
        device,
        model_path,
        mix_type=config.get("mix_type"),
        alpha=config.get("alpha", 1.0),
        mix_prob=config.get("mix_prob", 1.0)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    cm_val = evaluate_predictions(model, val_loader, device)
    cm_test = evaluate_predictions(model, test_loader, device)

    return model, {
        "logs": logs,
        "val": confusion_stats(cm_val),
        "test": confusion_stats(cm_test)
    }


def run_all_experiments(search_space, n_samples=10, folder=None):
    if folder is None:
        results_dir = RESULTS_DIR
    else:
        results_dir = os.path.join(RESULTS_DIR, folder)

    os.makedirs(results_dir, exist_ok=True)

    exp_id = 0

    for base in random_search(search_space, n_samples):

        for model_name in search_space["model"]:
            config = {**base, "model": model_name}

            results = []
            for seed in SEEDS:
                path = os.path.join(results_dir, f"model_{exp_id}_{model_name}_{seed}.pth")
                model, res = run_single(config, seed, path)
                results.append(res)

            with open(os.path.join(results_dir, f"exp_{exp_id}_{model_name}.json"), "w") as f:
                json.dump({"config": config, "runs": results}, f, indent=2)

        exp_id += 1
