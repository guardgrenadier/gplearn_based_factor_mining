import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from customized_functions import *
from Functions import *
import statsmodels.api as sm
#from factor_production import stock_data_dict
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', 100)  # 设置最大显示列数
pd.set_option('display.width', 1000)       # 控制台宽度（避免换行）


file_zz1000 = 'data/zz1000index_14-24.csv'
df_zz1000 = pd.read_csv(file_zz1000, encoding='gbk')
df_zz1000['date'] = pd.to_datetime(df_zz1000['date'].astype(str), format='%Y%m%d')
df_zz1000 = df_zz1000.sort_values(by='date')


# 要测试的因子集
factor_test_set = ['sub(covariance_15(X3, X4), ma_20(X6))'
                   ]

for factor in factor_test_set:
    print(f'开始测试因子：{factor}')
    try:
        file_path = f'factor_{factor}.csv'
        df = pd.read_csv(file_path, encoding='gbk')
        df = df[['date', 'stock_code', 'factor', 'neutralized_factor', 'CLOSE', 'VWAP', 'RETURN']]

        calculate_ic(df, factor_col='neutralized_factor', start_date='2021-01-01', end_date='2024-12-31', period=5, close_col='CLOSE')

        res, relative, turnover_hist = quantile_test(df, df_zz1000, factor_col='neutralized_factor', n_quantiles=10,
                                                     start_date='2021-01-01', end_date='2024-12-31',
                                                     price_type='CLOSE', cost_method='turnover')

        report = generate_report(res, turnover_hist, factor)

    except Exception as e:
        print(f"处理表达式 [{factor}] 时发生异常: {str(e)}")
        continue  # 跳过当前表达式继续执行

