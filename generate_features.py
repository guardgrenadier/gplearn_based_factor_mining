import os
import pickle
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, spearmanr
import datetime
import inspect


matplotlib.use('TkAgg')  # 切换后端
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)  # 设置最大显示列数
pd.set_option('display.width', 1000)       # 控制台宽度（避免换行）


file = 'data/zz1000_14-24.csv'
df = pd.read_csv(file, encoding='gbk')
print(df.head(30))


### 数据清理和新特征生成
# 数据排序
df = df.sort_values(by=['stock_code', 'date'])
print(df)


# 1，生成新特征，包括收益率，5日换手率，5日成交量，5日收益率
def create_ts_features(group):  # 假设T日已知当日信息，即收盘后
    group['TURNOVER_5D'] = group['TURNOVER'].rolling(5).mean()
    group['VOLUME_5D'] = group['VOLUME'].rolling(5).sum()
    group['RETURN_5D'] = (group['CLOSE'] / group['CLOSE'].shift(5)) - 1
    group['PCT_CHG_5D'] = (group['CLOSE'].shift(-6) / group['CLOSE'].shift(-1)) - 1
    group['RETURN'] = (group['CLOSE'] / group['CLOSE'].shift(1)) - 1
    return group


# 应用分组计算
df = df.groupby('stock_code', group_keys=False).apply(create_ts_features)
print(df.head(30))
print(df.tail(30))


# 检查无缺失值
for column in df.columns:
    print(column)
    print(df[df[column].isna()])


# 检查0值
for column in df.columns:
    print(column)
    print(df[df[column]==0])


# 保存数据
os.makedirs('data', exist_ok=True)
df.to_csv('zz1000_with_new_feat.csv', index=False, encoding='gbk')


