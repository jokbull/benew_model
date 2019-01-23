import numpy as np
from util.decorator import cache, np_cache
from collider.utils.logger import system_log


@np_cache
def benew_rank(return_expectation):
    # system_log.info("rank: %s" % np.nansum(return_expectation))
    i = 0
    rank_num = np.full(len(return_expectation), np.nan)
    v0, s0 = np.unique(return_expectation, return_index=True)  # 这里不直接用rank/order之类，是为了解决因子值有相同的情况
    for v, s in zip(v0, s0):
        if np.isnan(v):
            pass
        else:
            rank_num[return_expectation == v] = i
            i += 1

    rank_num /= i - 1
    rank_num[rank_num == 1] -= 1e-8
    return rank_num


@cache()
def calculate_rank_mean(**kwargs):
    return_rank = benew_rank(-kwargs['stock_return'])
    holding_weight = kwargs['holding_weight']
    return return_rank[holding_weight > 0].mean()


@cache()
def calculate_rank_weighted_mean(**kwargs):
    return_rank = benew_rank(-kwargs['stock_return'])
    holding_weight = kwargs['holding_weight']
    return np.nansum(return_rank * holding_weight)


@cache()
def calculate_rank_cumsum(**kwargs):
    return_rank = benew_rank(-kwargs['stock_return'])
    holding_weight = kwargs['holding_weight']
    return sum(holding_weight[holding_weight > 0][return_rank[holding_weight > 0] < kwargs["threshold"]])


@cache()
def calculate_exposure_weighted_mean(**kwargs):
    exposure = kwargs.get("exposure")
    holding_weight = kwargs.get("holding_weight")
    hedge_weight = kwargs.get("hedge_weight")
    return np.dot(np.nan_to_num(exposure), np.nan_to_num(holding_weight - hedge_weight))


# 利用直线插值，计算x2y和y2x
def x_to_y(x, cum):
    y0 = np.insert(cum, 0, 0)
    x0 = np.arange(0, len(cum) + 1) / len(cum)
    index = np.sum(x0 < x)
    return (x - x0[index - 1]) / (x0[index] - x0[index - 1]) * (y0[index] - y0[index - 1]) + y0[index - 1]


def y_to_x(y, cum):
    y0 = np.insert(cum, 0, 0)
    x0 = np.arange(0, len(cum) + 1) / len(cum)
    index = np.sum(y0 < y)
    return (y - y0[index - 1]) / (y0[index] - y0[index - 1]) * (x0[index] - x0[index - 1]) + x0[index - 1]


@cache()
def calculate_exposure_CDF(**kwargs):
    exposure = kwargs.pop("exposure")

    # 如果是多个因子, 那么递归，单因子计算
    if len(exposure.shape) > 1 and exposure.shape[0] > 1:
        return [calculate_exposure_CDF(exposure=expo, **kwargs) for expo in exposure]

    # 以下计算CDF指标
    holding_weight = kwargs.get("holding_weight")
    hedge_weight = kwargs.get("hedge_weight")

    long_side = holding_weight > 0
    short_side = hedge_weight > 0

    # 排序, cumsum
    # 需要减少精度，不然会出现1<1的问题
    a = np.argsort(exposure[long_side])
    cumWeightL = np.round(np.cumsum(holding_weight[long_side][a]), 8)

    b = np.argsort(exposure[short_side])
    cumWeightS = np.round(np.cumsum(hedge_weight[short_side][b]), 8)

    # HARD-CODE 分100份
    # FIXME: 为什么要abs?
    return 2 * sum(
        [0.01 * abs(j - x_to_y(x=y_to_x(y=j, cum=cumWeightS), cum=cumWeightL)) for j in np.linspace(0, 1, 100)])
