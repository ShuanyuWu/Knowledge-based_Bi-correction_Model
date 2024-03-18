from pytorch_lightning.callbacks import EarlyStopping
import torch
import numpy as np

my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.01,
    mode='min'
)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, version, dataset):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, version, dataset)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, version, dataset)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, version, dataset):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + version + '_' + dataset + '_' + f'vali_loss={val_loss:.3f}.pkl')
        torch.save(model.state_dict(), path + version + '_' + dataset + '_' + 'best_model.pkl')
        self.val_loss_min = val_loss