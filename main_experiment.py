import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import *
from utils.set_seed import seed_torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from utils.metrics import metric, EPERR
from model.KbBcM import KBBCM

class LagFreeForecasting:
    def __init__(self, path, split_date, input_dim, output_dim):
        pass

    def validate(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

if __name__ == '__main__':
    # 不同数据集主要差异是dataset类中的self.feature dim和tssa中的相似度拼接结果
    hdc = LagFreeForecasting('./data/WDD_11_15_tssa.csv', [2013, 12, 31], 7, 1)
    hdc.fit()
    hdc.predict()