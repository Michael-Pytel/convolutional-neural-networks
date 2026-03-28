import os

WORKSPACE_DIR = "/workspace"
DATA_DIR = "/tmp/cinic10"
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")

EPOCHS = 1000

SEEDS = [0,1,2]

DEVICE = "cuda"

EARLY_STOPPING_PATIENCE = 5