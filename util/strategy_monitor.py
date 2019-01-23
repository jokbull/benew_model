import pandas as pd
import os
from common.trade_date import generate_trade_date_sequence
from common.data_source_from_bundle import load_data_from_npy
from common.data_source_from_pkl import load_data_from_pkl
from common.strategy import read_holding
from util.strategy_feature import *


def holding_feature(trade_date, strategy_path: str, features=[], hedge_index="weight_index_500", **kwargs):
    """
    计算持仓的特征
    :param trade_date:
    :param strategy_path:
    :param features:
    :param hedge_index:
    :param kwargs:
    :return:
    """
    trade_date_array = generate_trade_date_sequence(trade_date)
    if len(trade_date_array) > 1:
        result = pd.concat(
            [holding_feature(date, strategy_path, features, hedge_index, **kwargs) for date in trade_date_array]
        )
    else:
        factors = kwargs.get("factors", [])

        trade_date_array = trade_date_array[0]
        # step 1. read factors exposure
        if len(factors) > 0:
            kwargs["exposure"] = load_data_from_npy(trade_date_array, factors)

        # step 1. read predicted_stock_return
        if "stock_return_name" not in kwargs:
            # 从pkl文件读predicted_stock_return, kwargs中需要包括root, scenario
            kwargs["stock_return"] = load_data_from_pkl(trade_date_array, "predicted_stock_return", **kwargs)
        else:
            # 从bundle读predicted_stock_return
            kwargs["stock_return"] = load_data_from_npy(trade_date_array, [kwargs.get("stock_return_name")])

        # load pool if necessary
        if "pool_name" in kwargs:
            kwargs["pool"] = load_data_from_npy(trade_date_array, kwargs.get("pool_name"))

        # step 2. read strategy holding
        kwargs["holding_weight"] = read_holding(trade_date_array, strategy_path, offset=0, return_type="numpy")

        if hedge_index is not None:
            hedge_weight_raw = load_data_from_npy(trade_date_array, hedge_index)
            kwargs["hedge_weight"] = np.nan_to_num(hedge_weight_raw) / np.nansum(hedge_weight_raw)
        else:
            kwargs["hedge_weight"] = None

        # step 4. 计算特征
        result = []
        for fun, name, args in features:
            items = name if isinstance(name, list) else [name]
            result = result + [pd.DataFrame({
                "trade_date": trade_date_array,
                "strategy": os.path.basename(strategy_path),
                "item": items,
                "value": fun(trade_date=trade_date_array, item=items, **kwargs, **args)
            })]
        result = pd.concat(result)

    return result


def calculate_attribution(**kwargs):
    """
    归因
    :param kwargs:
    :return:
    """
    pass


if __name__ == "__main__":
    factors = ["style_beta_2", "style_size_2"]

    _calculate_rank_features = [
        (calculate_rank_mean, "rank_mean", {}),
        (calculate_rank_weighted_mean, "rank_weighted_mean", {}),
        (calculate_rank_cumsum, "rank_cumsum_10", {'threshold': 0.1}),
        (calculate_rank_cumsum, "rank_cumsum_20", {'threshold': 0.2}),
        (calculate_rank_cumsum, "rank_cumsum_30", {'threshold': 0.3}),
        (calculate_rank_cumsum, "rank_cumsum_40", {'threshold': 0.4}),
        (calculate_rank_cumsum, "rank_cumsum_50", {'threshold': 0.5})
    ]

    _calculate_exposure_features = [
        (calculate_exposure_weighted_mean, ["exposure_" + fac for fac in factors], {}),
        (calculate_exposure_CDF, ["exposure_CDF_" + fac for fac in factors], {})
    ]

    d = holding_feature(("20190102", "20190104"), "/Volumes/会牛/策略管理/benew/Simulation/swing6f",
                        features=_calculate_exposure_features + _calculate_rank_features,
                        stock_return_name="predicted_stock_return_f1",
                        factors=factors,
                        strategy="swing6f",
                        )
    # d.to_csv("test.csv")
    print(d)

    # attribution = holding_feature(trade_date=("20190102", "20190104"),
    #                               strategy_path="/Volumes/会牛/策略管理/benew/Simulation/swing6f",
    #                               features=[(calculate_attribution,
    #                                          ["attr_" + _ for _ in factors],
    #                                          {})],
    #                               stock_return="forward_return_1_close_f1",
    #                               pool="pool_01_final_f1",
    #                               factor=factors)

