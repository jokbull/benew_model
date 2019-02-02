from common.data_source_from_bundle import load_data_from_npy
import numpy as np
import pandas as pd


def calculate_factor_correlation(factors, dates, pool_name, **kwargs):
    if isinstance(dates, list) or isinstance(dates, np.ndarray):
        corr_matrix = [calculate_factor_correlation(factors, date) for date in dates]
        return corr_matrix
    else:
        pool = load_data_from_npy(dates, pool_name) == 1
        data = load_data_from_npy(dates, factors, return_type="dict")
        df = pd.DataFrame({k: v[pool] for k, v in data.items()})
        corr_matrix = df.corr()
        return corr_matrix


if __name__ == "__main__":
    dates = "20181228"
    factors = [
        "benew_p06_TOP_20181031_tp_0935_1000_1109143553641_after",
        "benew_p01_noma_tvalue_20180801_0809185250191_after",
        "benew_p02_TOP_TMEAN_20181031_tp_0935_1000_1123003001628_after",
        "benew_p06_TOP_UPPER_20181031_tp_0935_1000_1115190448740_after",
        "benew_p06_TOP_TMEAN_UPPER_20181031_tp_0935_1000_1203020650541_after",
        "benew_p02_TOP_20181031_tp_0935_1000_1113135033072_after",
        "benew_p06_TOP_TMEAN_20181031_tp_0935_1000_1126220252804_after",
        "benew_p02_TOP_TMEAN_20181031_tp_0935_1000_1122224638628_after",
        "benew_p06_WTOP_R1017_20181031_tp_0935_1000_1112090325399_after",
        "benew_p06_noma_tvalue_20180928_tp_0935_1000_1008130743764_after",
        "benew_p02_TOP_TMEAN_20181031_tp_0935_1000_1127164607929_after",
        "benew_p02_TOP_TMEAN_20181031_tp_0935_1000_1128215207463_after",
        "benew_p1_noma_tvalue_20180702_0722124734704_after",
        "benew_p1_noma_tvalue_20180702_0721111235788_after",
        "benew_p02_TOP_20181031_tp_0935_1000_1113045429763_after"
    ]
    result = calculate_factor_correlation(factors, dates, pool_name="pool_01")
    print(result)
