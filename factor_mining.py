import pickle
import gplearn
import matplotlib
from gplearn.genetic import SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from scipy.sparse import csr_matrix, hstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, spearmanr
import datetime
import inspect
#from Functions import *
from sklearn.preprocessing import StandardScaler
from customized_functions import function_set
from joblib import Memory


matplotlib.use('TkAgg')  # 切换后端
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)  # 设置最大显示列数
pd.set_option('display.width', 1000)       # 控制台宽度（避免换行）


file = 'data/zz1000_with_new_feat.csv'
df = pd.read_csv(file, encoding='gbk')

df = df.sort_values(by=['stock_code', 'date'])  # 排序


# ------------------ 配置SymbolicRegressor ------------------
# 选定使用的函数
function_basic = ['add', 'sub', 'sqrt', 'log', 'inv', 'neg']
function_all = function_basic + function_set

# 选择训练区间
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] > pd.to_datetime('2022-01-01')) & (df['date'] <= pd.to_datetime('2023-12-31'))]


# 选择参与训练的特征
features_used = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'RETURN', 'VOLUME', 'VWAP']
df = df.dropna(subset=['RETURN', 'PCT_CHG_5D'])


# 生成传入SymbolicRegressor的X，Y
X_features = df[features_used].values
np.save('X_features.npy', X_features, allow_pickle=False)
X_features = np.load('X_features.npy')

Y_train = df[['PCT_CHG_5D']].values.ravel()


# ------------------ 生成每行数据对应的信息(X_aux)，用于适应度计算 ------------------
# 日期、股票代码、行业类别（中信一级）、市值、5日收益率、换手率、成交量
X_aux = df[['date', 'stock_code', 'industry', 'MKT_CAP', 'RETURN_5D', 'TURNOVER_5D', 'VOLUME_5D']].values


# ------------------ 自定义的适应度函数，计算5天RankIC均值 ------------------
def rank_ic_metric(y, y_pred, sample_weight, aux_data):
    dates = aux_data[:, 0]
    # 定义存储因子的dataframe
    temp = pd.DataFrame({
        'date': dates,
        'stock_code': aux_data[:, 1],
        'industry': aux_data[:, 2],
        'MKT_CAP': aux_data[:, 3].astype(float),
        'RETURN_5D': aux_data[:, 4].astype(float),
        'TURNOVER_5D': aux_data[:, 5].astype(float),
        'VOLUME_5D': aux_data[:, 6].astype(float),
        'factor': y_pred,
        'target': y,
        'weight': sample_weight
    })

    temp['MKT_CAP'] = pd.to_numeric(temp['MKT_CAP'], errors='coerce')
    temp['RETURN_5D'] = pd.to_numeric(temp['RETURN_5D'], errors='coerce')
    temp['TURNOVER_5D'] = pd.to_numeric(temp['TURNOVER_5D'], errors='coerce')
    temp['VOLUME_5D'] = pd.to_numeric(temp['VOLUME_5D'], errors='coerce')
    del aux_data

    # 检查因子值是否正常，若不正常直接返回适应度-1
    factor = temp['factor'].values
    if np.isnan(factor).any():
        return -1
    if np.max(factor) - np.min(factor) < 1e-8:
        return -1
    # ------------------ 对因子进行中位数去极值、中性化和标准化操作，注意是在每个截面上进行 ------------------
    rank_ic_list = []
    temp = temp.dropna(subset=['RETURN_5D', 'VOLUME_5D', 'TURNOVER_5D'])
    for date, group in temp.groupby('date'):
        factor = group['factor'].values
        # 中位数去极值
        median = np.median(factor)
        mad = np.median(np.abs(factor - median))
        factor_clipped = np.clip(factor, median - 5 * mad, median + 5 * mad)

        X = pd.get_dummies(group['industry'], drop_first=True)
        X['MKT_CAP'] = np.log(group['MKT_CAP'])  # 对市值取对数
        X['RETURN_5D'] = group['RETURN_5D']
        X['VOLUME_5D'] = group['VOLUME_5D']
        X['TURNOVER_5D'] = group['TURNOVER_5D']

        model = LinearRegression()
        model.fit(X, factor_clipped)
        residuals = factor_clipped - model.predict(X)  # 残差即为中性化之后的因子

        # 因子标准化
        std = residuals.std()
        if std < 1e-8:
            rank_ic_list.append(-1)
            continue
        else:
            residuals = (residuals - residuals.mean()) / std

        # Rank_IC计算
        ic = spearmanr(residuals, group['target']).statistic
        if not np.isnan(ic):
            rank_ic_list.append(ic)

    del temp
    return np.mean(rank_ic_list)


gp = SymbolicRegressor(generations=3,
                       population_size=100,
                       tournament_size=10,
                       low_memory=True,
                       metric=make_fitness(function=rank_ic_metric, greater_is_better=True),
                       stopping_criteria=1.0,
                       function_set=function_all,
                       const_range=None,
#                      parsimony_coefficient='auto',
                       init_depth=(2, 5),
                       p_crossover=0.35,
                       p_subtree_mutation=0.15,
                       p_hoist_mutation=0.145,
                       p_point_mutation=0.1,
                       p_point_replace=0.25,
                       verbose=1,
                       n_jobs=8,
                       random_state=11)

gp.fit(X_features, Y_train, aux_data=X_aux)

# ------------------ 打印结果 ------------------
# 获取最终代所有个体
last_generation = gp._programs[-1]

sorted_programs = sorted(last_generation,
                         key=lambda x: x.fitness_,
                         reverse=True)

# 提取前5名表达式
top5_expressions = [program.__str__() for program in sorted_programs[:5]]

# 打印结果
for j, expr in enumerate(top5_expressions):
    print(f"Rank {j} (Fitness={sorted_programs[j].fitness_:.4f}):")
    print(expr)
    print("-" * 50)

