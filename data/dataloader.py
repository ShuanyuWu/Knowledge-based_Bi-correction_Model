import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

feature_scaler, target_scaler = StandardScaler(), StandardScaler()
tssa_score_list, mult_score_list, addi_score_list = [], [], []


class DATASET(Dataset):
    def __init__(self, path, dataset, mode, seq_len):
        self.path = path + dataset + '.csv'
        self.dataset = dataset
        self.window_size = seq_len
        mode_map = {'train': 0, 'test': 1, 'val': 2}
        self.mode = mode_map[mode]

        self._read_data()

    def _read_data(self):
        df = pd.read_csv(self.path, delimiter=',')
        self.feature_dim = df.shape[1] - 2  # get number of the columns (exclude the date column and the OT/WD column)
        if self.dataset == 'wd_daily':
            self.split_date = 1096 - 8
            feature = feature_scaler.fit_transform(
                np.array(df[df.columns[2:]]).reshape(-1, self.feature_dim))  # 全部数据都可作为输入，同时去掉最后一位数据，避免数据泄露
            target = target_scaler.fit_transform(np.array(df['WD']).reshape(-1, 1))  # 将水量数据做一步错位，避免数据泄露
        elif self.dataset == 'ecl_daily':
            self.split_date = 767  # split number of ecl in comparison.py
            feature = feature_scaler.fit_transform(
                np.array(df[df.columns[2:]]).reshape(-1, self.feature_dim))  # 全部数据都可作为输入，同时去掉最后一位数据，避免数据泄露
            target = target_scaler.fit_transform(np.array(df['OT']).reshape(-1, 1))  # 将水量数据做一步错位，避免数据泄露
        elif self.dataset == 'exchange_rate_daily':
            self.split_date = 5311  # split number of exchange rate in comparison.py
            feature = feature_scaler.fit_transform(
                np.array(df[df.columns[2:]]).reshape(-1, self.feature_dim))  # 全部数据都可作为输入，同时去掉最后一位数据，避免数据泄露
            target = target_scaler.fit_transform(np.array(df['OT']).reshape(-1, 1))  # 将水量数据做一步错位，避免数据泄露
        else:
            print('Dataset not available')

        self.x_train = feature[:self.split_date, :]
        self.x_test = feature[self.split_date:, :]

        self.y_train = target[: self.split_date]  # 此处不用截短，可在训练循环过程中进行操作
        self.y_test = target[self.split_date:]

    def __getitem__(self, item):
        # 注意滑窗设计
        # 注意不要有信息泄露
        if self.mode == 0:
            return self.x_train[item: item+self.window_size, :], self.y_train[item+self.window_size]
        elif self.mode == 1:
            return self.x_test[item: item+self.window_size, :], self.y_test[item+self.window_size]

    def __len__(self):
        if self.mode == 0:
            return len(self.x_train) - self.window_size
        elif self.mode == 1:
            return len(self.x_test) - self.window_size

    def inverse_transform(self, data):
        return target_scaler.inverse_transform(data)


def dataset_scaler():
    return feature_scaler, target_scaler