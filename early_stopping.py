import torch


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0, path=None):

        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

            if self.path is not None:
                torch.save(model.state_dict(), self.path)

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True