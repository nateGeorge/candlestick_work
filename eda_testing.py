import sys

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../stock_prediction/code')
import dl_quandl_EOD as dlq

stocks = dlq.load_stocks()

qqq = stocks['QQQ']
for i in range(1, 20):
    qqq['Close_{}d_pct_chg'.format(i)] = qqq['Adj_Close'].pct_change(i) * 100
    qqq['Close_{}d_pct_chg_future'.format(i)] = qqq['Close_{}d_pct_chg'.format(i)].shift(-i)


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

def plot_future_pct_chg(df, days=1):
    """
    plots future percent change 'days' number of days in the future
    """
    col = 'Close_{}d_pct_chg_future'.format(days)
    df[col].hist(bins=50)
    mean_str = str(round(df[col].mean(), 1))
    std_str = str(round(df[col].std(), 1))
    plt.title('average: ' + mean_str + '\nstddev: ' + std_str)
    plt.xlabel('future {}d close-close pct change'.format(days))
    plt.ylabel('count')
    plt.show()
