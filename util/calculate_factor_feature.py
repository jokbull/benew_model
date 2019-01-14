import numpy as np
from scipy.stats import spearmanr
from collider.data.base_data_source import BaseDataSource
from common.configure import read_configure

bundle_path = read_configure()['bundle_path']

DataSource = BaseDataSource()
DataSource.initialize(bundle_path)

td = DataSource.trading_dates


def load_data_from_npy(trade_date, factor_name):
    return DataSource.get_bar(trade_date, [factor_name])[factor_name]


def calculate_factor_feature(factors, forward_return_name, pool_name, dates, func, **kwargs) -> np.ndarray:
    if isinstance(dates, list) or isinstance(dates, np.ndarray):
        # rawIC = np.array([calculate_IC(factors, forward_return_name, pool_name, date) for date in dates])
        rawIC = np.array([calculate_factor_feature(factors, forward_return_name, pool_name, date, func, **kwargs) for date in dates])
        return rawIC
    else:
        pool = load_data_from_npy(dates, pool_name) == 1
        forward_return = load_data_from_npy(dates, forward_return_name)
        rawIC = np.array([func(load_data_from_npy(dates, f), forward_return, pool, **kwargs) for f in factors])
        return rawIC


def calculate_IC(a, b, pool, **kwargs):

    try:
        direction = kwargs.get("direction", 1)
        return spearmanr(direction * a[pool], b[pool], nan_policy="omit")[0]
    except Exception as e:
        print(e)



import statsmodels.api as sm
def calculate_factor_return(a, b, pool, **kwargs):
    direction = kwargs.get("direction", 1)
    model = sm.OLS(b[pool], direction * a[pool], hasconst=False, missing="drop")

    result = model.fit()  # method = "qr"
    return result.params[0]

def calculate_tvalue(a, b, pool, **kwargs):
    direction = kwargs.get("direction", 1)
    model = sm.OLS(b[pool], direction * a[pool], hasconst=False, missing="drop")

    result = model.fit()  # method = "qr"
    return result.tvalues[0]


if __name__ == "__main__":
    start_date = "20180701"
    end_date = "20181226"
    factors = [
               # "benew_p5_ma10_hist_tvalue_p06_20180531_t7_0606221539248_after",
               # "benew_p5_ma10_0322001055_after",
               # "benew_p5_ma20_0326083326_after",
               # "benew_p05_noma_tvalue_20150601_0816225040769_after",
               # "benew_p06_noma_tvalue_20180901_tp_0935_1000_0922191517773_after",
               # "benew_p06_noma_tvalue_20180901_tp_0935_1000_0922221054686_after",
               # "benew_p02_noma_tvalue_20180901_tp_0935_1000_0920194834095_after",
               # "benew_p5_ma10_hist_kaleido_p05_20180515_0530181708_after",
               # "benew_p06_noma_tvalue_20180928_tp_0935_1000_1001183211111_after",
               # "benew_p05_noma_tvalue_20180901_tp_0935_1000_0904104741953_after",
               # "benew_p02_noma_tvalue_20180901_tp_0935_1000_0923223647237_after",
               # "benew_p02_WTOP_R1011_20180928_tp_0935_1000_1017133635924_after",
               "predicted_stock_return_f1",
               "flow_estimation_fitted_f1"
               ]
    pool_name = "pool_01_final_f1"
    forward_return_name = "forward_return_5_f1"

    dates = td.get_trading_dates(start_date, end_date)
    # IC = calculate_factor_feature(factors, forward_return_name, pool_name, dates, calculate_IC, direction=-1)
    import pandas as pd

    # df = pd.DataFrame(IC, index=dates, columns=factors)
    # print(df)
    # result = calculate_factor_feature(factors, forward_return_name, pool_name, dates, calculate_factor_return, direction=-1)
    result = calculate_factor_feature(factors, forward_return_name, pool_name, dates, calculate_IC,
                                      direction=1)

    df = pd.DataFrame(result, index=dates, columns=factors)
    df.to_csv("test_IC.csv")

