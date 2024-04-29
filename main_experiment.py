import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
from torch import nn
from time import *
from torch.utils.data import DataLoader
from utils.metrics import metric
from data.dataloader import DATASET, feature_scaler, target_scaler
from model.KbBcM import MainModule

# seed_torch(42)

class HDC:
    def __init__(self, path, dataset, input_dim, output_dim):
        self.batch_size = 1
        self.train_data = DataLoader(
            DATASET(path, dataset, 'train', seq_len=7),  # seq_len = input_dim, 且此时batch_size要为1，这样才能将数据全部规整给t_lstm
            batch_size=self.batch_size,
            drop_last=True
        )
        self.test_data = DataLoader(
            DATASET(path, dataset, 'test', seq_len=7),
            batch_size=self.batch_size,
            drop_last=True
        )
        self.dataset = dataset
        self.device = torch.device('cuda:0')
        self.model = MainModule(input_dim, 7, 1, 1+1, 24, output_dim, self.batch_size)  # 整个模型的输入、输出，分别是l-lstm和s-lstm的输入、输出。
        self.model.to(self.device)
        self.epochs = 100
        self.early_stop = 5

    def _mwmc(self, error):
        mean = np.mean(error)
        grade = 5
        numerater = np.zeros([len(error) - 1, grade])
        denomiter = np.zeros(len(error))

        r = np.zeros(grade)
        w = np.zeros(grade)
        # 先计算sum((xi - x_mean)2)
        for j in range(len(error)):
            denomiter[j] = math.pow((error[j] - mean), 2)
        for i in range(1, grade + 1):  # (1-5)i用以表示阶数尺度，j用以表示error长度尺度
            for j in range(len(error) - i):
                numerater[j, i - 1] = (error[j] - mean) * (error[j + i] - mean)
        # print('分子与分母分别为：\n{}, \n{}'.format(numerater, denomiter))
        for i in range(grade):
            r[i] = sum(numerater[:, i]) / sum(denomiter)
        # print('各阶自相关系数为：{}'.format(r))
        # 各阶自相关系数归一化（确保之后加权求得的值不会大于1）
        r = abs(r)  # 求权重时，自相关系数取绝对值
        for i in range(grade):
            w[i] = r[i] / sum(r)
        # print('权重系数为：{}'.format(w))

        state = np.zeros(len(error))
        # 状态划分
        # 此处注意加mask，针对状态转移概率矩阵中的某些单元，进行选择性屏蔽。
        for i in range(len(error)):
            if error[i] <= -0.1:
                state[i] = -2
            elif -0.1 < error[i] < -0.01:
                state[i] = -1
            elif -0.01 <= error[i] <= 0.01:
                state[i] = 0
            elif 0.01 < error[i] < 0.1:
                state[i] = 1
            elif error[i] >= 0.1:
                state[i] = 2

        # 计算加权马尔可夫转移矩阵
        count = np.zeros([grade, grade, grade])  # 第三个维度表示各阶自相关系数所对应的概率转移矩阵
        # 初步统计转移数目
        for i in range(1, grade + 1):  # 与之前一样，大循环看各阶的值
            for j in range(len(error) - i):  # i表示马尔科夫阶数，j表示从当前时刻开始进行转移，j+i即表示从当前时刻转移i步对应的状态转移情况
                if state[j] == -2:
                    if state[j + i] == -2:
                        count[i - 1, 0, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 0, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 0, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 0, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 0, 4] += 1
                elif state[j] == -1:
                    if state[j + i] == -2:
                        count[i - 1, 1, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 1, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 1, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 1, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 1, 4] += 1
                elif state[j] == 0:
                    if state[j + i] == -2:
                        count[i - 1, 2, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 2, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 2, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 2, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 2, 4] += 1
                elif state[j] == 1:
                    if state[j + i] == -2:
                        count[i - 1, 3, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 3, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 3, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 3, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 3, 4] += 1
                elif state[j] == 2:
                    if state[j + i] == -2:
                        count[i - 1, 4, 0] += 1
                    elif state[j + i] == -1:
                        count[i - 1, 4, 1] += 1
                    elif state[j + i] == 0:
                        count[i - 1, 4, 2] += 1
                    elif state[j + i] == 1:
                        count[i - 1, 4, 3] += 1
                    elif state[j + i] == 2:
                        count[i - 1, 4, 4] += 1
        # print('状态统计结果为：\n{}\n'.format(count))
        # 计算转移概率
        prob = np.zeros([grade, grade, grade])
        distribute_prob = np.zeros([grade, grade])
        weighted_prob = np.zeros(grade)
        # 计算各阶状态转移概率矩阵
        for m in range(grade):
            for i in range(grade):
                if sum(count[m, i, :]) != 0:  # 排除分母为0的可能性
                    for j in range(grade):
                        prob[m, i, j] = count[m, i, j] / sum(count[m, i, :])
        # print('状态转移概率计算结果为：\n{}\n'.format(prob))
        # 计算加权后的状态转移概率矩阵
        for m in range(1, grade + 1):  # 从步长为1开始算
            # print(state[len(error) - m])
            if state[len(error) - m] == -2:
                distribute_prob[m - 1, :] = prob[m - 1, 0, :]
            elif state[len(error) - m] == -1:
                distribute_prob[m - 1, :] = prob[m - 1, 1, :]
            elif state[len(error) - m] == 0:
                distribute_prob[m - 1, :] = prob[m - 1, 2, :]
            elif state[len(error) - m] == 1:
                distribute_prob[m - 1, :] = prob[m - 1, 3, :]
            elif state[len(error) - m] == 2:
                distribute_prob[m - 1, :] = prob[m - 1, 4, :]
        # print(distribute_prob)
        for i in range(grade):
            for j in range(grade):
                weighted_prob[i] = weighted_prob[i] + (w[i] * distribute_prob[j, i])
        # print('加权状态转移概率为：\n{}\n'.format(weighted_prob))
        index = np.argmax(weighted_prob)
        cor_error = 0
        if index == 0:
            cor_error = -0.2
        elif index == 1:
            cor_error = -0.05
        elif index == 2:
            cor_error = 0.00
        elif index == 3:
            cor_error = 0.05
        elif index == 4:
            cor_error = 0.2
        return cor_error


    def fit(self):
        # 初始化
        criterion = nn.MSELoss()
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train_loss_memory = 1
        early_stop_count = 0
        # 训练TLSTM
        for epoch in range(self.epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time()
            for i, data in enumerate(self.train_data):
                iter_count += 1
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float().to(self.device)
                model_optimizer.zero_grad()
                model_out = self.model(batch_x)
                model_loss = criterion(model_out, batch_y)
                model_loss.backward()
                model_optimizer.step()
                train_loss.append(model_loss.detach().cpu().numpy())

            print('epoch {}，MSE Loss：{}'.format(epoch, np.average(np.array(train_loss))))

            if np.average(np.array(train_loss)) <= train_loss_memory:
                torch.save(self.model.state_dict(), './results/pkl/HDC_{}.pkl'.format(self.dataset))
                train_loss_memory = np.average(np.array(train_loss))
                early_stop_count = 0  # 每次能够重新下降都对early count清零
                print("Loss decrease，Saving the model")
            else:
                early_stop_count += 1
                print("Loss not decrease，drop the model")
                if early_stop_count > self.early_stop:
                    break

    def predict(self):
        model = self.model
        model.load_state_dict(torch.load('./results/pkl/HDC_{}.pkl'.format(self.dataset)))
        model.eval()

        iter_count = 0
        preds, trues = [], []

        for i, data in enumerate(self.test_data):
            iter_count += 1
            batch_x = data[0].float().to(self.device)
            batch_y = data[1].float().to(self.device)

            # 网络前向传播
            model_out = model(batch_x).detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(model_out)
            trues.append(true)

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)

        preds = preds[1:]
        trues = trues[1:]

        errors = np.zeros(len(preds))

        for j in range(0, len(preds)):
            # errors[j] = ((trues[j] - preds[j]) - (trues[j] - naive[j])) / preds[j]
            errors[j] = (preds[j] - trues[j]) / trues[j]
            if j >= 7:
                Marcov_cor_out = self._mwmc(errors[(j - 7): j])
                preds[j] = preds[j] * (1 - Marcov_cor_out)

        preds = target_scaler.inverse_transform(preds.reshape(-1, 1))
        trues = target_scaler.inverse_transform(trues.reshape(-1, 1))
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}\n'.format(rmse, mse, mae, mape))


        # df_out = pd.DataFrame(columns=['HDC', 'True'])
        # df_out['HDC'] = preds.flatten()
        # df_out['True'] = trues.flatten()
        # df_out.to_csv('./results/data/experiment_{}_{}.csv'.format(self.dataset, round(mape, 2)), index=False)
        #
        # plt.plot(preds, label='preds')
        # plt.plot(trues, label='true')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    # 不同数据集主要差异是dataset类中的self.feature dim和tssa中的相似度拼接结果
    dataset = ['wd_daily', 'ecl_daily', 'exchange_rate_daily']
    for j in range(10):
        for i, data in enumerate(dataset):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            input_dim = data  # 用来指示不同数据集，并在之中切换
            output_dim = 1
            hdc = HDC('./data/', data, 7, output_dim)
            hdc.fit()
            hdc.predict()
