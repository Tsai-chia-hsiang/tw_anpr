import random
import numpy as np
import torch

def reproducible(seed:int = 891122):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure reproducibility with CUDA
    torch.cuda.manual_seed_all(seed)

    # Configure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False