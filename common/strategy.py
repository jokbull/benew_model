import os
import numpy  as np
import pandas as pd
from common.data_source_from_bundle import __td__, __ds__


def dataframe_to_ndarray(df):
    """
    pd.DataFrame to ndarray, 除去trade_date, wind_code, 其他n列变成n*4000的ndarray
    :param df: 数据
    :return: ndarray
    """
    columns = df.columns
    assert ("wind_code" in columns)
    result = pd.merge(pd.DataFrame({'wind_code': __ds__.codes}), df, on=['wind_code'], how="left")
    res = np.r_[[np.array(result[col]) for col in columns if col not in ["wind_code", "trade_date"]]]
    if res.shape[0] == 1:
        return res.ravel()
    else:
        return res


def ndarray_to_dataframe(array, **kwargs):
    """
    ndarray to pd.DataFrame
    :param array: 数据
    :param kwargs: column_name = str
    :return: pd.DataFrame
    """
    return pd.DataFrame({'wind_code': __ds__.codes, kwargs.get("column_name", "value"): array})


def read_holding(trade_date, strategy_path, offset=0, return_type="numpy"):
    """
    读持仓数据
    :param trade_date:
    :param strategy_path:
    :param offset: 持仓日期的偏移，offset=1为昨日
    :param return_type: numpy or pandas
    :return:
    """
    date = __td__.get_previous_trading_date(trade_date, offset)
    holding_path = os.path.join(strategy_path, "holding", "%s.csv" % date)

    # 为了确保中文路径也可以读
    with open(holding_path, "r") as infile:
        holding_df = pd.read_csv(infile)

    holding_array = dataframe_to_ndarray(holding_df)
    # price
    price = __ds__.get_bar(date, ["close_n"])["close_n"]

    weight = np.nan_to_num(price * holding_array)
    weight /= np.sum(weight)

    if return_type == "numpy":
        return weight
    elif return_type == "pandas":
        result = ndarray_to_dataframe(weight, column_name="weight")
        result['trade_date'] = date
        return result
    else:
        raise NotImplementedError("not support return_type %s" % return_type)
