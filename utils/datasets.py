import numpy as np
from torch.utils.data import Dataset

class WindDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        target = self.y[idx].astype(np.float32)

        return x, target