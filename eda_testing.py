import sys

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../stock_prediction/code')
import dl_quandl_EOD as dlq

stocks = dlq.load_stocks()

qqq = stocks['QQQ']
qqq['Close_1d_pct_chg'] = qqq['Adj_Close'].pct_change() * 100

def get_ohlc_for_talib(df):
    """
    gets open high low close data for talib functions
    """
    open_price, high_price, low_price, close_price = df['Adj_Open'].values, df['Adj_High'].values, df['Adj_Low'].values, df['Adj_Close'].values
    return open_price, high_price, low_price, close_price

o, h, l, c = get_ohlc_for_talib(qqq)

# get all candlestick pattern functions
# all functions take open, high, low, close, except
# abandoned baby, dark cloud cover, evening doji star, evening star, mat hold, morning doji star, morning star, which all have
# penetration=0
# penetration is percentage of penetration of one candle into another, e.g. 0.3 is 30%
pattern_functions = talib.get_function_groups()['Pattern Recognition']
column_names = [p[3:] for p in pattern_functions] # for dataframe; cuts out CDL
# for bearish, seems to give -100, for nothing, 0, for bullish, 100
data = {}
for col, p in zip(column_names, pattern_functions):
    data[col] = getattr(talib, p)(o, h, l, c)

df = pd.DataFrame(data)
df.index = qqq.index
full_df = pd.concat([qqq, df], axis=1)

# current day is beltholt, longline, and marubozu (bearish)
# get all indices of where this happened

filtered = full_df[(full_df['BELTHOLD'] == -100) & (full_df['LONGLINE'] == -100) & (full_df['MARUBOZU'] == -100)].copy()
next_days = [full_df.index.get_loc(f) + 1 for f in filtered.index[:-1]]  # don't use latest because that was today

next_days_df = full_df.iloc[next_days]

next_days_df['Close_1d_pct_chg'].hist(bins=50)
mean_str = str(round(next_days_df['Close_1d_pct_chg'].mean(), 1))
std_str = str(round(next_days_df['Close_1d_pct_chg'].std(), 1))
plt.title('average: ' + mean_str + '\nstddev: ' + std_str)
plt.xlabel('close-close pct change')
plt.ylabel('count')
plt.show()
