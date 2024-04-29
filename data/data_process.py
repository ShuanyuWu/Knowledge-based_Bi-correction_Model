import pandas as pd
import numpy as np

# resample the houly ecl into daily
ecl_original = pd.read_csv('electricity.csv')
ecl_original['date'] = pd.to_datetime(ecl_original['date'])
ecl_original.set_index('date', inplace=True)
# print(ecl_original)
daily_ecl = ecl_original.resample('D').sum()
daily_ecl = daily_ecl[['OT', 'OT', '0', '1', '2', '3', '4', '5', '6', '315', '316', '317', '318', '319']]
daily_ecl.to_csv('ecl_daily.csv', index=True)
# print(daily_ecl)

# set time index for exchange rate
exchange_original = pd.read_csv('exchange_rate_daily.csv')
exchange_original.set_index('date', inplace=True)
exchange = exchange_original[['OT', 'OT', '0', '1', '2', '3', '4', '5', '6']]
exchange.to_csv('exchange_rate_daily.csv', index=True)

# rearrange the columns of wd_daily
wd_original = pd.read_csv('wd_daily.csv')
wd_original['date'] = pd.to_datetime(wd_original['date'])
wd_original.set_index('date', inplace=True)
wd = wd_original[['WD', 'WD', 'MinT', 'MaxT']]
wd.to_csv('wd_daily.csv', index=True)

# determine the split date
ecl = pd.read_csv('ecl_daily.csv')
exchange = pd.read_csv('exchange_rate_daily.csv')
split_date_ecl = ecl.iloc[int(0.7 * len(ecl)), 0]
split_date_exchange = exchange.iloc[int(0.7 * len(exchange)), 0]
print('split number:\nECL:{}, Exchange rate:{}'.format(int(0.7 * len(ecl)), int(0.7 * len(exchange))))
print(split_date_ecl, split_date_exchange)
