import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_log_error

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return mean_absolute_error(true, pred)

def MSE(pred, true):
    return mean_squared_error(true, pred)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
    # return mean_squared_error(true, pred)

def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.sqrt(sum(true ** 2))

def MAPE(pred, true):
    return 100 * mean_absolute_percentage_error(true, pred)

def SMAPE(pred, true):
    smape = np.mean(
        np.abs(pred - true) / ((np.abs(pred) + np.abs(true)) / 2)
    )
    return smape

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def ND(pred, true):
    return sum(abs(pred - true)) / sum(abs(true))

def DS(pred, true):
    count = 0
    pred = np.array(pred)
    true = np.array(true)
    pred_delta = np.empty(len(pred) - 1)
    true_delta = np.empty(len(true) - 1)

    for i in range(len(pred) - 1):
        pred_delta[i] = pred[i + 1] - pred[i]
        true_delta[i] = true[i + 1] - true[i]
        if pred_delta[i] * true_delta[i] >= 0:  # 有错
            count += 1
        else:
            pass
    return 100 * count / len(true_delta)

def MPM(pred, true):
    count_tu = 0
    count_td = 0
    count_fu = 0
    count_fd = 0
    pred = np.array(pred)
    true = np.array(true)
    pred_delta = np.empty(len(pred) - 1)
    true_delta = np.empty(len(true) - 1)
    for i in range(len(pred) - 1):
        pred_delta[i] = pred[i + 1] - pred[i]
        true_delta[i] = true[i + 1] - true[i]
        if true_delta[i] >= 0:  # 有错
            if pred_delta[i] >= 0:
                count_tu += 1
            else:
                count_fu += 1
        else:
            if pred_delta[i] >= 0:
                count_fd += 1
            else:
                count_td += 1
    return (count_td + count_tu) / (count_fd + count_fu + count_tu + count_td)

def EPERR(pred, true):
    # print(np.abs(np.array(true[:-1] - np.array(true[1:]))) * np.abs(true[1:] - pred[1:]) / (np.abs(true[1:] - pred[1:])))
    # return np.mean((np.abs(np.array(true[:-1] - np.array(true[1:]))) * np.abs(true[1:] - pred[1:])) / np.abs(true[1:] - pred[1:]))
    a = np.array(true[:-1])
    b = np.array(true[1:])
    c = np.array(pred[1:])
    # print(abs(b - c) - abs(b - a))
    # _ = (abs(b - a) - abs(b - c)) / np.maximum(np.abs(b - c), 0.001)
    _ = (abs(b - a) - abs(b - c))
    output_errors = np.mean(_)
    # return (DS(pred, true) / 100 * len(a)) * output_errors  # 左边一项反映趋势一致性

    # 第二种算法,
    count = 0
    pred_delta = np.empty(len(a) - 1)
    true_delta = np.empty(len(a) - 1)
    for i in range(len(a) - 1):
        pred_delta[i] = c[i + 1] - c[i]
        true_delta[i] = b[i + 1] - b[i]
        if pred_delta[i] * true_delta[i] >= 0:
            count += (abs(b[i] - a[i]) - abs(b[i] - c[i]))
        else:
            pass
    return count / len(a)



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = r2_score(true, pred)
    ds = DS(pred, true)
    # msle = mean_squared_log_error(true, pred)
    
    return mae, mse, rmse, mape, mspe

def metric_kbbcm(pred, true):
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    mape = MAPE(pred, true)
    r2 = r2_score(true, pred)
    ds = DS(pred, true)
    nd = ND(pred, true)
    mspe = MSPE(pred, true)
    mpm = MPM(pred, true)
    eperr = EPERR(pred, true)
    return rmse, mape, ds, mpm, eperr