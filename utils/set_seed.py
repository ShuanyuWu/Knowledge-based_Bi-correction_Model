import random
import os
import numpy as np
import torch


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 以下两个对精度提升很小，可不考虑
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
