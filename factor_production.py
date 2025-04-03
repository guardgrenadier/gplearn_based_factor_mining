import os
from sklearn.linear_model import LinearRegression
import pickle
import gplearn
import matplotlib
from gplearn.genetic import SymbolicRegressor
import numpy as np
import pandas as pd
from Functions import *


matplotlib.use('TkAgg')  # 切换后端
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)  # 设置最大显示列数
pd.set_option('display.width', 1000)       # 控制台宽度（避免换行）


file = 'data/zz1000_with_new_feat.csv'
df = pd.read_csv(file, encoding='gbk', parse_dates=['date'])
df = df.dropna(subset=['RETURN'])

# 数据字典
stock_data_dict = {'X0': df['OPEN'].values, 'X1': df['HIGH'].values, 'X2': df['LOW'].values, 'X3': df['CLOSE'].values,
                   'X4': df['RETURN'].values, 'X5': df['VOLUME'].values, 'X6': df['VWAP'].values}


# 要生成的因子表达式集合
factor_expression_set = ['sub(covariance_15(X3, X4), ma_20(X6))'
                         ]

for expression in factor_expression_set:
    try:
        # 为每个因子创建文件夹
        os.makedirs(f'因子{expression}', exist_ok=True)

        temp = df.copy()
        temp['neutralized_factor'] = np.nan

        # 计算因子
        factor = parse_and_compute(expression, stock_data_dict)
        temp['factor'] = factor

        columns_keep = ['date', 'stock_code', 'stock_name', 'factor', 'neutralized_factor', 'CLOSE', 'VWAP',
                        'MKT_CAP', 'industry', 'TURNOVER_5D', 'VOLUME_5D', 'RETURN_5D', 'PCT_CHG_5D', 'RETURN']
        temp = temp[columns_keep].dropna(subset=['RETURN_5D', 'VOLUME_5D', 'TURNOVER_5D'])

        # ------------------ 对因子进行中位数去极值、中性化和标准化操作（在每个截面上进行） ------------------
        for date, group in temp.groupby('date'):
            group = group.sort_values(by=['stock_code'])
            factor = group['factor'].values
            
            # 中位数去极值
            median = np.median(factor)
            mad = np.median(np.abs(factor - median))
            factor_clipped = np.clip(factor, median - 5 * mad, median + 5 * mad)

            # 中性化处理
            factor_clipped = np.clip(factor_clipped, -1e8, 1e8)  # 防止中性化前溢出
            X = pd.get_dummies(group['industry'], drop_first=True)
            X['MKT_CAP'] = np.log(group['MKT_CAP'])  # 对市值取对数
            X['RETURN_5D'] = group['RETURN_5D']
            X['VOLUME_5D'] = group['VOLUME_5D']
            X['TURNOVER_5D'] = group['TURNOVER_5D']

            model = LinearRegression()
            model.fit(X, factor_clipped)
            residuals = factor_clipped - model.predict(X)

            # 标准化
            std = residuals.std()
            residuals = (residuals - residuals.mean()) / std
            temp.loc[group.index, 'neutralized_factor'] = residuals

        factor_data = temp[['date', 'stock_code', 'factor', 'neutralized_factor', 'CLOSE', 'VWAP', 'RETURN']]
        # 删去每个股票前20天的因子
        factor_data = factor_data.groupby('stock_code').apply(lambda group: group.iloc[20:]).reset_index(drop=True)
        factor_data = factor_data.sort_values(by=['stock_code', 'date'])

        # 保存因子
        factor_data.to_csv(os.path.join(f'因子{expression}', f'factor_{expression}.csv'), encoding='gbk')

        # 中性化检查
        check_results = check_neutralization(temp, feature_cols=['VOLUME_5D', 'TURNOVER_5D', 'RETURN_5D'])
        print(f'处理因子{expression}成功')

    except Exception as e:
        print(f"处理表达式 [{expression}] 时发生异常: {str(e)}")
        continue  # 跳过当前表达式继续执行
