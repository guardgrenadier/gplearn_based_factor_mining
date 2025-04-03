import os
import pickle
import gplearn
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
from customized_functions import *
from gplearn.functions import sqrt1,neg1,log1,inv1,abs1
import statsmodels.api as sm


# ------------------ 生成函数字典 ------------------
function_dict = {}
function_sources = [delta_functions,delay_functions,signedpower_functions,ma_functions,decaylinear_functions,ts_min_functions,
                    ts_max_functions,ts_argmin_functions,ts_argmax_functions,ts_rank_functions,ts_sum_functions,
                    ts_product_functions,ts_stddev_functions,correlation_functions,covariance_functions
                    ]

# 遍历每个字典，将其中的函数加入 function_dict
for func_source in function_sources:
    function_dict.update(func_source)

# 添加非时间窗口函数
function_dict.update({
    'add': np.add,
    'sub': np.subtract,
    'mul': safe_mul,
    'div': safe_div,
    'sqrt': sqrt1,
    'neg': neg1,
    'log': log1,
    'inv': inv1,
    'abs': abs1
})

# 检查函数导入
print("Available functions in function_dict:")
print(list(function_dict.keys()))


def parse_and_compute(expression, data_dict):
    """递归解析因子表达式并计算因子"""
    # 去掉外层空格
    expression = expression.strip()
    # 如果是变量，返回值
    if expression in data_dict:
        return data_dict[expression]

    # 如果是函数，调用
    if '(' in expression and expression.endswith(')'):
        # 找到函数名和括号内的内容
        func_name = expression[:expression.index('(')]
        args_str = expression[expression.index('(') + 1: -1]
        # 解析函数的参数
        args = []
        bracket_count = 0
        current_arg = []
        for char in args_str:
            if char == ',' and bracket_count == 0:
                # 参数分隔
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
                if char == '(':
                    bracket_count += 1
                elif char == ')':
                    bracket_count -= 1
        # 添加最后一个参数
        if current_arg:
            args.append(''.join(current_arg).strip())

        # 递归计算参数值
        computed_args = [parse_and_compute(arg, data_dict) for arg in args]

        # 调用对应函数
        if func_name in function_dict:
            return function_dict[func_name](*computed_args)
        else:
            raise ValueError(f"Unknown function: {func_name}")

    # 报错：未知表达式格式
    raise ValueError(f"Invalid expression: {expression}")


def quantile_test(factordata, benchmarkdata, factor_col=None, n_quantiles=10, holding_days=5, cost_rate=0.003,
                  start_date=None, end_date=None, price_type='CLOSE', cost_method='fixed'):
    """
    单因子测试——分层回测，支持固定和按换手率计算交易费用
    cost_method = 'fixed' or 'turnover'
    """
    # ------------------ 日期范围过滤 ------------------
    factordata['date'] = pd.to_datetime(factordata['date'])
    benchmarkdata['date'] = pd.to_datetime(benchmarkdata['date'])

    if start_date:
        factordata = factordata[factordata['date'] >= pd.to_datetime(start_date)]
        benchmarkdata = benchmarkdata[benchmarkdata['date'] >= pd.to_datetime(start_date)]
    if end_date:
        factordata = factordata[factordata['date'] <= pd.to_datetime(end_date)]
        benchmarkdata = benchmarkdata[benchmarkdata['date'] <= pd.to_datetime(end_date)]

    # ------------------ 价格数据预处理 ------------------
    price_pivot = factordata.set_index(['date', 'stock_code'])[price_type]
    benchmark_pivot = benchmarkdata.set_index(['date', 'stock_code'])['CLOSE']

    # ------------------ 初始化组合净值 ------------------
    dates = factordata['date'].unique()
    portfolio = pd.DataFrame(np.nan,
                             index=pd.Index(dates, name='date'),
                             columns=[f'Quantile_{q}' for q in range(1, n_quantiles + 1)] + ['benchmark'])
    portfolio.iloc[0, :] = 1.0  # 组合初始净值为1
    relative_value = portfolio.drop('benchmark', axis=1)  # 相对净值组合

    prev_holdings = {}  # 用于换手率计算
    portfolio_turnover = relative_value.copy()

    # ------------------ 分层测试循环 ------------------
    factor_by_date = factordata.groupby('date')  # 因子按日期分组

    for i in tqdm(range(0, len(dates) - holding_days, holding_days)):
        rebalance_date = dates[i]
        next_rebalance = dates[i + holding_days] if (i + holding_days) < len(dates) else dates[-1]
        buy_date = dates[i + 1] if (i + 1) < len(dates) else dates[-1]
        next_buy = dates[i + holding_days + 1] if (i + holding_days + 1) < len(dates) else dates[-1]

        # 更新基准净值
        if rebalance_date in benchmark_pivot.index:
            benchmark_ret = (benchmark_pivot.loc[next_buy] / benchmark_pivot.loc[buy_date] - 1).values
            portfolio.loc[next_rebalance, 'benchmark'] = portfolio.loc[rebalance_date, 'benchmark'] * (1 + benchmark_ret)

        # ------------------ 更新组合净值 ------------------
        try:
            # 获得当期因子值
            current_factor = factor_by_date.get_group(rebalance_date).copy()
            # 分层
            current_factor['quantile'] = pd.qcut(current_factor[factor_col], n_quantiles, labels=False) + 1

            # 计算收益率
            buy_price = price_pivot.loc[buy_date]
            sell_price = price_pivot.loc[next_buy]
            valid_stocks = buy_price.index.intersection(sell_price.index)
            returns = (sell_price[valid_stocks] / buy_price[valid_stocks] - 1).rename('return')
            merged = current_factor.merge(returns, left_on='stock_code', right_index=True, how='inner')

            # 交易费用计算
            if cost_method == 'turnover':  # 按换手率计算费率
                turnover = calculate_turnover(current_factor, prev_holdings, n_quantiles)
                cost = turnover * cost_rate

                # 记录换手率
                for q in turnover.index:
                    portfolio_turnover.loc[next_rebalance, f'Quantile_{q}'] = turnover[q]

            else:
                cost = cost_rate  # 固定费率

            # 计算每组收益率，扣除交易费用
            quantile_ret = merged.groupby('quantile')['return'].mean() - cost

            # 更新净值
            for q, ret in quantile_ret.items():
                portfolio.loc[next_rebalance, f'Quantile_{q}'] = portfolio.loc[rebalance_date, f'Quantile_{q}'] * (1 + ret)

            # 更新持仓记录
            prev_holdings[rebalance_date] = current_factor.set_index('stock_code')['quantile']

        except (KeyError, ValueError) as e:
            print(f"发生异常: {str(e)}")
            continue

    # ------------------ 计算相对收益 ------------------
    benchmark_nav = portfolio['benchmark'].values
    for col in relative_value.columns:
        relative_value[col] = portfolio[col].values / benchmark_nav

    return portfolio, relative_value, portfolio_turnover


def calculate_turnover(current_factor, prev_holdings, n_quantiles):
    """计算每层换手率"""
    if not prev_holdings:
        return pd.Series(1.0, index=range(1, n_quantiles + 1))

    # 获取最近持仓分位数据
    prev_date = sorted(prev_holdings.keys())[-1]

    prev_q = prev_holdings[prev_date].to_frame()

    # 合并当前持仓（需包含分位信息）
    current_q = current_factor[['stock_code', 'quantile']].set_index('stock_code')
    merged = prev_q.join(current_q, how='outer', lsuffix='_prev', rsuffix='_current')

    # 初始化换手率容器
    turnover_dict = {q: 0.0 for q in range(1, n_quantiles + 1)}

    # 遍历每个历史分位计算卖出换手
    for q_prev, group in merged.groupby('quantile_prev'):
        sold_stocks = group[group['quantile_current'].isna()].shape[0]
        prev_count = group.shape[0]
        turnover_dict[q_prev] += sold_stocks / (2 * prev_count)

    # 遍历每个当前分位计算买入换手
    for q_current, group in merged.groupby('quantile_current'):
        bought_stocks = group[group['quantile_prev'].isna()].shape[0]
        current_count = group.shape[0]
        turnover_dict[q_current] += bought_stocks / (2 * current_count)

    return pd.Series(turnover_dict)


def generate_report(portfolio, turnover, factor_expression):
    """生成分层测试的绩效分析"""
    report = pd.DataFrame()
    portfolio.index = pd.to_datetime(portfolio.index).sort_values()
    # 净值数据
    portfolio_nav = portfolio.copy().dropna().astype('float')
    # 换手率数据
    portfolio_turnover = turnover.copy().dropna().astype('float')
    # 计算收益率
    portfolio_returns = portfolio_nav.pct_change().dropna()
    # 计算月度收益率
    portfolio_monthly_ret = portfolio_nav.resample('ME').last().pct_change().dropna()

    for col in portfolio.columns:
        nav = portfolio_nav[col]
        returns = portfolio_returns[col]
        monthly_ret = portfolio_monthly_ret[col]

        # -------------------------- 组合评价指标计算 --------------------------
        start_date = nav.index[0]
        end_date = nav.index[-1]

        # 年化收益率
        total_years = (end_date - start_date).days / 365
        annualized_return = (nav.iloc[-1] / nav.iloc[0]) ** (1 / total_years) - 1

        # 年化波动率（使用交易日标准差）
        annualized_volatility = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率为0）
        sharpe_ratio = (annualized_return - 0) / annualized_volatility

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (cummax - nav) / cummax
        max_drawdown = drawdown.max()

        # 月度胜率
        monthly_win_rate = (monthly_ret > portfolio_monthly_ret['benchmark']).mean()

        # 平均双边换手率
        turnover = portfolio_turnover[col].mean() if col != 'benchmark' else 0

        # -------------------------- 保存指标到报告 --------------------------
        report.loc[col, '年化收益率'] = f'{annualized_return:.2%}'
        report.loc[col, '年化波动率'] = f'{annualized_volatility:.2%}'
        report.loc[col, '夏普比率'] = f'{sharpe_ratio:.2f}'
        report.loc[col, '最大回撤'] = f'{max_drawdown:.2%}'
        report.loc[col, '月度胜率'] = f'{monthly_win_rate:.2%}'
        report.loc[col, '平均双边换手率'] = f'{turnover:.2%}'

    # -------------------------- 绘制净值曲线 --------------------------
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    handles, labels = [], []

    for col in portfolio_nav.columns:

        line, = ax1.plot(portfolio_nav[col], linewidth=1, label=col)
        handles.append(line)
        labels.append(col)

    # 设置图表元素
    ax1.set_title(f'因子{factor_expression}分层组合净值表现', fontsize=14)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('净值', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 统一图例
    ax1.legend(handles, labels, loc='lower left', fontsize=10)

    plt.savefig(os.path.join(f'因子{factor_expression}', f'quantile_test_{factor_expression}.png'))
    plt.close()

    # -------------------------- 组合评价可视化 --------------------------
    # 创建表格图像
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    table = ax.table(
        cellText=report.values,
        colLabels=report.columns,
        rowLabels=report.index,
        cellLoc='center',
        loc='center'
    )

    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.5)  # 调整表格尺寸

    plt.title(f'因子 {factor_expression} 绩效报告', fontsize=14)
    plt.savefig(os.path.join(f'因子{factor_expression}', f'report_table_{factor_expression}.png'), bbox_inches='tight')
    plt.close()

    return print(report)


def calculate_ic(factor_with_returns, period=1, start_date=None, end_date=None, factor_col='neutralized_factor', close_col='CLOSE'):
    """计算因子值与period天后的收益率的Rank_IC"""
    df = factor_with_returns[['date', 'stock_code', close_col, factor_col]].copy()
    # 计算目标——period日后收益率
    df['target'] = df.groupby('stock_code')[close_col].transform(lambda x: x.shift(-period - 1) / x.shift(-1) - 1)
    df = df.dropna(subset=['target'])

    # 日期范围过滤
    df['date'] = pd.to_datetime(df['date'])
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    ic_list = []
    for date, group in tqdm(df.groupby('date')):
        ic = spearmanr(group[factor_col], group['target']).statistic

        if not np.isnan(ic):
            ic_list.append(ic)

    # RankIC统计计算
    ic_array = np.array(ic_list)
    ic_mean = np.nanmean(ic_array)
    ic_std = np.nanstd(ic_array)
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    positive_pct = np.mean(ic_array > 0)

    print(f'Rank IC值序列均值为: {ic_mean:.4f}')
    print(f'Rank IC值序列标准差为: {ic_std:.4f}')
    print(f'IC_IR为: {ic_ir:.4f}' if ic_ir is not None else 'IC_IR为: NaN')
    print(f'Rank IC值序列大于零的占比为: {positive_pct:.2%}')

    return ic_list


def check_neutralization(factor, factor_col='neutralized_factor',
                         industry_col='industry', cap_col='MKT_CAP',
                         feature_cols=None):
    """因子中性化效果验证函数,返回一个包含检验结果的字典"""
    df = factor.copy()
    # 初始化结果存储字典
    results = {'industry': {}, 'cap': {}, 'features': {}}

    # 行业中性化检验
    industry_means = df.groupby(industry_col)[factor_col].mean()
    industry_std = industry_means.std()
    results['industry']['mean_std'] = industry_std

    # 市值中性化检验
    df['log_cap'] = np.log(df[cap_col])
    corr, pval = spearmanr(df['log_cap'], df[factor_col])
    results['cap']['spearmanr'] = corr
    results['cap']['pvalue'] = pval

    # 其他特征中性化检验
    for feat in feature_cols:
        corr, pval = spearmanr(df[feat], df[factor_col])
        results['features'][feat] = {'spearmanr': corr, 'pvalue': pval}

    # 可视化
    plt.figure(figsize=(15, 5))

    # 行业分布图
    plt.subplot(131)
    industry_means.plot(kind='bar', alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('行业因子均值（中信一级）')

    # 市值散点图
    plt.subplot(132)
    plt.scatter(df['log_cap'], df[factor_col], s=1, alpha=0.3)
    plt.xlabel('对数市值')
    plt.ylabel(factor_col)
    plt.title('市值相关性')
    print(results)
    # 特征相关系数
    plt.subplot(133)
    corr_values = [results['cap']['spearmanr']] + \
                  [v['spearmanr'] for v in results['features'].values()]
    labels = ['MKT_CAP'] + feature_cols
    plt.barh(labels, corr_values, alpha=0.6)
    plt.title('特征相关性')

    plt.tight_layout()
    plt.show()

    return results




