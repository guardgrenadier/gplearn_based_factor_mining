import gc
from functools import wraps
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from gplearn.functions import make_function
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def _check_invalid_input(arr, tolerance=1e-10):
    """检查自定义函数的输入，检查输入数组是否是存在全Nan/全0/常数"""
    # 检查全NaN
    if np.isnan(arr).all():
        return True
    # 检查全无穷
    if np.isinf(arr).all():
        return True
    # 检查常数值（跳过NaN和无穷值）
    finite_vals = arr[np.isfinite(arr)]
    if np.nanmax(finite_vals) - np.nanmin(finite_vals) < tolerance:
        return True
    return False


def _protected_div(x1, x2):
    """受保护的除法，防止数值溢出"""
    with np.errstate(divide='ignore', invalid='ignore'):
        if _check_invalid_input(x1) or _check_invalid_input(x2):
            return np.zeros_like(x1)
        return np.where(np.abs(x2) > 1e-8, x1/x2, x1/0.1)
safe_div = make_function(function=_protected_div, name='div', arity=2)


def _protected_multiply(x, y):
    """受保护的乘法，防止数值溢出"""
    if _check_invalid_input(x) or _check_invalid_input(y):
        return np.zeros_like(x)

    sign_x = np.sign(x)
    sign_y = np.sign(y)
    log_x = np.log(np.abs(x) + 1e-16)
    log_y = np.log(np.abs(y) + 1e-16)

    result = sign_x * sign_y * np.exp(log_x + log_y)
    # 硬截断，数值太大因子没意义
    max_abs = np.abs(result).max()
    if max_abs > 1e40:
        return np.zeros_like(x)
    return result
safe_mul = make_function(function=_protected_multiply, name='mul', arity=2)



def create_delay_function(d):
    def _delay(x):
        """时序滞后d天"""
        result = np.zeros_like(x, dtype=float)
        result[d:] = x[:-d]
        return result
    return make_function(function=_delay, name=f'delay_{d}', arity=1)
# 滞后窗口
delays = [1, 2, 3, 5, 10, 20]
delay_functions = {f'delay_{d}': create_delay_function(d) for d in delays}


def create_corr_function(d):
    def _correlation(x, y):
        """计算两个特征间在最近d日内相关系数，分块处理"""
        if _check_invalid_input(x) or _check_invalid_input(y):
            return np.zeros_like(x)
        result = np.zeros_like(x, dtype=float)

        n = len(x)
        L = min(400000, n)
        for block_idx in range(0, n, L):
            chunk_start = max(0, block_idx - d + 1)
            chunk_end = min(n, block_idx + L)
            x_chunk = x[chunk_start:chunk_end]
            y_chunk = y[chunk_start:chunk_end]
            # 向量化
            x_win = sliding_window_view(x_chunk, d)
            y_win = sliding_window_view(y_chunk, d)
            # 计算
            x_std = np.std(x_win, axis=1, ddof=1)
            y_std = np.std(y_win, axis=1, ddof=1)
            constant_mask = (x_std < 1e-8) | (y_std < 1e-8)  # 检测标准差，若其中一个序列的窗口接近常数，则令相关系数为0
            mean_x = np.mean(x_win, axis=1)
            mean_y = np.mean(y_win, axis=1)
            cov = np.sum((x_win - mean_x[:, None]) * (y_win - mean_y[:, None]),
                         axis=1) / (d - 1)

            # 安全计算相关系数
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = x_std * y_std
                corr = np.divide(cov, denominator, out=np.zeros_like(cov),
                                 where=(~constant_mask) & (denominator > 1e-10))
            corr[constant_mask] = 0.0

            # 写入结果
            if block_idx == 0:
                result[d-1:chunk_end] = corr[:]
            else:
                result[block_idx:chunk_end] = corr[:]
            del x_win, y_win, x_chunk, y_chunk, mean_x, mean_y, corr

        return np.clip(result, -1.0, 1.0)
    return make_function(function=_correlation, name=f'correlation_{d}', arity=2)
windows = [5, 10, 15, 20]
correlation_functions = {f'correlation_{d}': create_corr_function(d) for d in windows}


def create_cov_function(d):
    def _covariance(x, y):
        """计算两个特征间在最近d日内协方差，分块处理"""
        if _check_invalid_input(x) or _check_invalid_input(y):
            return np.zeros_like(x)
        result = np.zeros_like(x, dtype=float)

        n = len(x)
        L = min(400000, n)
        for block_idx in range(0, n, L):
            chunk_start = max(0, block_idx - d + 1)
            chunk_end = min(n, block_idx + L)
            x_chunk = x[chunk_start:chunk_end]
            y_chunk = y[chunk_start:chunk_end]
            x_win = sliding_window_view(x_chunk, d)
            y_win = sliding_window_view(y_chunk, d)

            mean_x = np.mean(x_win, axis=1)
            mean_y = np.mean(y_win, axis=1)
            cov = np.sum((x_win - mean_x[:, None]) * (y_win - mean_y[:, None]), axis=1) / (d - 1)

            # 写入结果
            if block_idx == 0:
                result[d-1:chunk_end] = cov[:]
            else:
                result[block_idx:chunk_end] = cov[:]
            del x_win, y_win, x_chunk, y_chunk, mean_x, mean_y, cov

        return result
    return make_function(function=_covariance, name=f'covariance_{d}', arity=2)
covariance_functions = {f'covariance_{d}': create_cov_function(d) for d in windows}


# ------------------ Delta函数 ------------------
def create_delta_function(d):
    def _delta(x):
        """计算特征相对d日前变化量"""
        shifted = np.concatenate([np.zeros(d, dtype=float), x[:-d]])
        return x - shifted
    return make_function(function=_delta, name=f'delta_{d}', arity=1)
delta_functions = {f'delta_{d}': create_delta_function(d) for d in delays}


def create_signedpower_function(a):
    def _signedpower(x):
        """对特征进行有符号的幂运算"""
        if _check_invalid_input(x):
            return np.zeros_like(x)

        # 明确0值的符号
        sign_x = np.where(x == 0, 1.0, np.sign(x))
        # 硬截断
        upper_bound = 1e20
        abs_x = np.abs(x)
        safe_abs = np.clip(abs_x, 0, upper_bound)

        with np.errstate(over='ignore', invalid='ignore'):
            result = sign_x * np.power(safe_abs, a)

        # 二次防护
        result = np.where(np.abs(result) > upper_bound, sign_x * upper_bound, result)
        return result
    return make_function(function=_signedpower, name=f'signedpower_{a}', arity=1)
powers = [0.1, 0.25, 0.5, 1.5, 2.0]
signedpower_functions = {f'signedpower_{a}': create_signedpower_function(a) for a in powers}


def create_decaylinear_function(d):
    def _decay_linear(x):
        """对特征进行线性衰减计算"""
        result = np.zeros_like(x, dtype=float)
        # 计算权重
        weights = np.arange(1, d+1)[::-1]
        weights = weights / weights.sum()

        n = len(x)
        L = min(400000, n)
        for block_idx in range(0, n, L):
            chunk_start = max(0, block_idx - d + 1)
            chunk_end = min(n, block_idx + L)
            x_chunk = x[chunk_start:chunk_end]

            x_win = sliding_window_view(x_chunk, d)

            decaylinear = np.dot(x_win, weights)

            # 写入结果
            if block_idx == 0:
                result[d-1:chunk_end] = decaylinear[:]
            else:
                result[block_idx:chunk_end] = decaylinear[:]
            del x_win, x_chunk, decaylinear
        return result
    return make_function(function=_decay_linear, name=f'decay_linear_{d}', arity=1)
decay_delays = [2, 3, 5, 10, 15, 20]
decaylinear_functions = {f'decay_linear_{d}': create_decaylinear_function(d) for d in decay_delays}


def create_ma_function(d):
    def _ma(x):
        """对特征进行移动平均"""
        result = np.zeros_like(x, dtype=np.float64)

        n = len(x)
        L = min(400000, n)
        for block_idx in range(0, n, L):
            chunk_start = max(0, block_idx - d + 1)
            chunk_end = min(n, block_idx + L)
            x_chunk = x[chunk_start:chunk_end]

            x_win = sliding_window_view(x_chunk, d)

            ma = np.mean(x_win, axis=1)

            # 写入结果
            if block_idx == 0:
                result[d-1:chunk_end] = ma[:]
            else:
                result[block_idx:chunk_end] = ma[:]
            del x_win, x_chunk, ma
        return result
    return make_function(function=_ma, name=f'ma_{d}', arity=1)
ma_windows = [3, 5, 10, 20]
ma_functions = {f'ma_{d}': create_ma_function(d) for d in ma_windows}


def create_tsrank_function(d):
    def _ts_rank(x):
        """对特征计算过去d日内分位数"""
        result = np.zeros_like(x, dtype=float)

        n = len(x)
        L = min(400000, n)

        for block_idx in range(0, n, L):
            chunk_start = max(0, block_idx - d + 1)
            chunk_end = min(n, block_idx + L)
            x_chunk = x[chunk_start:chunk_end]

            x_win = sliding_window_view(x_chunk, d)

            ranks = rankdata(x_win, method='average', axis=1)[:, -1] / d

            # 写入结果
            if block_idx == 0:
                result[d-1:chunk_end] = ranks[:]
            else:
                result[block_idx:chunk_end] = ranks[:]
            del x_win, x_chunk, ranks
        return result
    return make_function(function=_ts_rank, name=f'ts_rank_{d}', arity=1)
ts_rank_functions = {f'ts_rank_{d}': create_tsrank_function(d) for d in [5, 10, 15, 20]}


# ------------------ 一些时间序列统计函数 ------------------
# ts_min, ts_max, ts_argmin, ts_argmax, ts_product, ts_sum, ts_stddev
def _create_ts_function(func, name):
    def wrapper(d):
        def _ts_func(x):
            result = np.zeros_like(x, dtype=float)

            n = len(x)
            L = min(400000, n)
            for block_idx in range(0, n, L):
                chunk_start = max(0, block_idx - d + 1)
                chunk_end = min(n, block_idx + L)
                x_chunk = x[chunk_start:chunk_end]

                x_win = sliding_window_view(x_chunk, d)

                ts_calc = func(x_win, axis=1)

                # 写入结果
                if block_idx == 0:
                    result[d-1:chunk_end] = ts_calc[:]
                else:
                    result[block_idx:chunk_end] = ts_calc[:]
                del x_win, x_chunk, ts_calc
            return result
        return make_function(function=_ts_func, name=f'{name}_{d}', arity=1)
    return wrapper

# 批量创建统计函数
ts_min_functions = {f'ts_min_{d}': _create_ts_function(np.min, 'ts_min')(d) for d in [3, 5, 10, 20]}
ts_max_functions = {f'ts_max_{d}': _create_ts_function(np.max, 'ts_max')(d) for d in [3, 5, 10, 20]}
ts_sum_functions = {f'ts_sum_{d}': _create_ts_function(np.sum, 'ts_sum')(d) for d in [2, 3, 5, 10, 20]}
ts_argmin_functions = {f'ts_argmin_{d}': _create_ts_function(np.argmin, 'ts_argmin')(d) for d in [3, 5, 10, 20]}
ts_argmax_functions = {f'ts_argmax_{d}': _create_ts_function(np.argmax, 'ts_argmax')(d) for d in [3, 5, 10, 20]}
ts_product_functions = {f'ts_product_{d}': _create_ts_function(np.prod, 'ts_product')(d) for d in [2, 3, 5, 10, 20]}
ts_stddev_functions = {f'ts_stddev_{d}': _create_ts_function(lambda x, axis: np.std(x, axis=axis, ddof=1), 'ts_stddev')(d) for d in [3, 5, 10, 20]}


# ------------------ 定义自定义函数集合 ------------------
function_set = [safe_mul, safe_div]
function_set.extend(delay_functions.values())
function_set.extend(correlation_functions.values())
function_set.extend(covariance_functions.values())
function_set.extend(delta_functions.values())
function_set.extend(signedpower_functions.values())
function_set.extend(decaylinear_functions.values())
function_set.extend(ma_functions.values())
function_set.extend(ts_min_functions.values())
function_set.extend(ts_max_functions.values())
function_set.extend(ts_argmin_functions.values())
function_set.extend(ts_argmax_functions.values())
function_set.extend(ts_rank_functions.values())
function_set.extend(ts_sum_functions.values())
function_set.extend(ts_product_functions.values())
function_set.extend(ts_stddev_functions.values())




