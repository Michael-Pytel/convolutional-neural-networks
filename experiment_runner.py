import os
import json
import random
import torch
import random
import math

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
            return 10 ** rng.uniform(math.log10(low), math.log10(high))
        elif scale == "linear":
            return rng.uniform(low, high)
        else:
            raise ValueError(f"Unknown scale: {scale}")

    return rng.choice(values)


def random_search(space, n_samples, seed=42):
    rng = random.Random(seed)

    models = space["model"]

    for model in models:
        for _ in range(n_samples):

            params = {
                "model": model,
                **{k: sample_param(rng, v) for k, v in space["common"].items()},
                **{k: sample_param(rng, v) for k, v in space[model].items()},
            }

            yield params

def run_single(config, seed, model_path):
    print(f"Seed: {seed}, Config: {json.dumps(config)}")
    set_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        config["batch_size"],
        use_augmentation=config.get("augmentation", True),
        model_name=config["model"],
        few_shot_k=config.get("few_shot_k", None),
    )

    model = get_model(config["model"], config.get("dropout", 0)).to(device)

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
        mixup_alpha=config.get("mixup_alpha", 1.0),
        cutmix_alpha=config.get("cutmix_alpha", 1.0),
        mix_prob=config.get("mix_prob", 1),
        p_mixup=config.get("p_mixup", 0.5)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    cm_val = evaluate_predictions(model, val_loader, device)
    cm_test = evaluate_predictions(model, test_loader, device)

    return model, {
        "logs": logs,
        "val": confusion_stats(cm_val),
        "test": confusion_stats(cm_test)
    }

def generate_configs(search_space=None, n_samples=10, configs_list=None, seed=42):
    if configs_list is not None:
        for cfg in configs_list:
            yield cfg

    elif search_space is not None:
        yield from random_search(search_space, n_samples, seed)

    else:
        raise ValueError("Provide either search_space or configs_list")

def run_all_experiments(search_space=None, n_samples=10, configs_list=None, folder=None):
    if folder is None:
        results_dir = RESULTS_DIR
    else:
        results_dir = os.path.join(RESULTS_DIR, folder)

    os.makedirs(results_dir, exist_ok=True)

    exp_id = 0

    for config in generate_configs(
        search_space=search_space,
        n_samples=n_samples,
        configs_list=configs_list
    ):

        results = []
        for seed in SEEDS:
            path = os.path.join(results_dir, f"model_{exp_id}_{config['model']}_{seed}.pth")
            model, res = run_single(config, seed, path)
            results.append(res)

        with open(os.path.join(results_dir, f"exp_{exp_id}_{config['model']}.json"), "w") as f:
            json.dump({"config": config, "runs": results}, f, indent=2)

        exp_id += 1
