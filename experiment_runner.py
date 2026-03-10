import os
import json
import random
from config import RESULTS_DIR, SEEDS, EPOCHS, DEVICE
from utils import set_seed
from datasets import get_dataloaders
from models import get_model
from train import train_model
from metrics import evaluate_predictions, confusion_stats
import itertools
import torch


device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


def random_search(search_space, n_samples, seed=None):
    keys = [k for k in search_space.keys() if k != "model"]
    all_combinations = list(itertools.product(*[search_space[k] for k in keys]))

    if seed is not None:
        random.seed(seed)

    random.shuffle(all_combinations)
    for combo in all_combinations[:n_samples]:
        yield dict(zip(keys, combo))


def run_all_experiments(search_space, n_samples=10):


    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_id = 0
    models = search_space["model"]

    for base_config in random_search(search_space, n_samples=n_samples, seed=42):
        print("Base hyperparameters:", base_config)

        for model_name in models:
            config = base_config.copy()
            config["model"] = model_name
            print("Running model:", model_name, "with config:", config)

            results = []

            for seed in SEEDS:
                print("Seed:", seed)
                
               
                model_path = os.path.join(
                    RESULTS_DIR,
                    f"best_model_exp{experiment_id}_{model_name}_seed{seed}.pth"
                )

                result = run_single(config, seed, model_path=model_path)
                results.append(result)

           
            path = os.path.join(
                RESULTS_DIR,
                f"experiment_{experiment_id}_{model_name}.json"
            )
            with open(path, "w") as f:
                json.dump({
                    "config": config,
                    "runs": results
                }, f)

        experiment_id += 1

def run_single(config, seed, model_path="best_model.pth"):
    set_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(config["batch_size"])

    model = get_model(config["model"], config["dropout"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    logs = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        EPOCHS,
        device,
        model_path=model_path
    )


    model.load_state_dict(torch.load(model_path))

    cm_val = evaluate_predictions(model, val_loader, device)
    cm_test = evaluate_predictions(model, test_loader, device)

    return {
        "logs": logs,
        "val_stats": confusion_stats(cm_val),
        "test_stats": confusion_stats(cm_test)
    }


