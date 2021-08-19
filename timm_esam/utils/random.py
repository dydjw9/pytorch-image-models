import random
import numpy as np
import torch


def random_seed(seed=42, rank=0,opt=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if opt > 0:
        random.seed(seed)
    else:
        random.seed(seed + rank)
