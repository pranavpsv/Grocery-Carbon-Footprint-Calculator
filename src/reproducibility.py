import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 0):
    """
    Sets all seeds for reproducibility

    Args:
        seed (int): The seed to fix

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

