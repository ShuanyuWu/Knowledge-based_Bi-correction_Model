import numpy as np
import torch
from torch import nn


class trend_extraction(nn.Module):
    def __init__(self):
        super(trend_extraction, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)  # 参数采用avg_pool的output_len计算得来

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class TLstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TLstm, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)
        self.trend = trend_extraction()

    def forward(self, x):
        res, trend = self.trend(x)
        x, t = self.rnn(x - res)  # 直接avg得到的res维度与原x不匹配，反操作下
        seq_len, batch_size, input_dim = x.shape
        x = x.view(seq_len * batch_size, input_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

class LnWeighting(nn.Module):
    def __init__(self):
        super(LnWeighting, self).__init__()

    def forward(self, x):
        return torch.tan(x) / torch.sum(torch.tan(x))


class TSSA(nn.Module):
    def __init__(self):
        super(TSSA, self).__init__()
        self.variance = nn.Parameter(torch.tensor(0.9), requires_grad=True)
        self.ln_weighting = LnWeighting()
        self.softmax_weighting = nn.Softmax(dim=0)  # 没错！e^{x}与x在0处相切，其在0-1的范畴内，增长速率更大，因此也更能区分出不同特征的相似性差异
        # self.score_function = torch.distributions.Normal(torch.tensor(0), self.variance)
        self.tss = []

    def _lpd(self, data_a, data_b):
        # 计算距离（相似性）
        # 序列梯度（delta value）计算
        gradient_a = torch.zeros(len(data_a) - 1)
        gradient_b = torch.zeros(len(data_a) - 1)
        for i in range(len(data_a) - 1):
            gradient_a[i] = data_a[i + 1] - data_a[i]
            gradient_b[i] = data_b[i + 1] - data_b[i]
        series_a = torch.zeros((len(data_a)))
        series_b = torch.zeros((len(data_b)))
        hpdd = torch.zeros(len(series_a))
        series_a[1:] = gradient_a
        series_b[1:] = gradient_b
        for i in range(len(series_a)):
            if series_a[i] * series_b[i] >= 0:
                hpdd[i] = series_a[i] - series_b[i]
            else:
                hpdd[i] = abs(series_a[i] - series_b[i])
        # grad_dis = sum(abs(series_a - series_b))
        return torch.sum(hpdd)

    def _gaussian_function(self, data, avg, sig):
        sqrt_2pi = torch.pow(torch.tensor(2 * np.pi).cuda(), torch.tensor(0.5).cuda())
        coef = torch.tensor(1.0).cuda() / (sqrt_2pi * sig)
        powercoef = torch.tensor(-1.0).cuda() / (torch.tensor(2).cuda() * torch.pow(sig, torch.tensor(2).cuda()))
        mypow = powercoef * (torch.pow((data - avg), torch.tensor(2).cuda()))
        return coef * (torch.exp(mypow))

    def forward(self, x):
        self.tss = []
        for i in range(1, x.size(1)):
            self.tss.append(self._lpd(x[:, 0].reshape(-1), x[:, i].reshape(-1)))
        tss = torch.tensor(self.tss).cuda()
        score = self._gaussian_function(tss, torch.tensor(0), self.variance)
        y = torch.sum(self.ln_weighting(score) * x[:, 1:]).reshape(-1)
        return y


class StackLSTM(nn.Module):
    # 初始化神经网络
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StackLSTM, self).__init__()
        self.transform = nn.Linear(input_size, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        self.reg = nn.Linear(hidden_size, output_size)
        # 最后再进行一次全连接，引入梯度变化量，同时采用drop out

    # 设置前向通路
    def forward(self, x):
        x, t = self.rnn(x)
        seq_len, batch_size, input_dim = x.shape
        x = x.view(seq_len * batch_size, input_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x


class MainModule(nn.Module):
    def __init__(self, input_dim1, hidden_dim1, output_dim1, input_dim2, hidden_dim2, output_dim2, batch_size):
        super(MainModule, self).__init__()
        # T-LSTM
        self.t_lstm = TLstm(input_dim1, hidden_dim1, output_dim1, 1)
        # TSSA
        self.tssa = TSSA()
        # Stack-LSTM
        self.s_lstm = StackLSTM(input_dim2, hidden_dim2, output_dim2, 5)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.batch_size = batch_size

    def forward(self, x):
        trend = self.t_lstm(x[0, :, 0].reshape(-1, self.batch_size, self.input_dim1))
        tssa = self.tssa(x[0, :, :])
        s_ltem_in = torch.cat([tssa.reshape(-1), trend.reshape(-1)])
        y = self.s_lstm(s_ltem_in.reshape(-1, self.batch_size, self.input_dim2))
        return y
