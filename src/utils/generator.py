from functools import wraps
from torch.utils.data import DataLoader

def generator(*args, **kwargs):
    data_loader = DataLoader(*args, **kwargs)
    while True:
        for batch in data_loader:
            yield batch['image'].numpy(), batch['label'].numpy()