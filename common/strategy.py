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


def get_last_file(path, date=None, key=None):
    """获取(小于等于指定日期)含有指定 key 的文件
    """
    files = []
    for fn in os.listdir(path):
        if key is not None:
            if fn.find(key) == -1:
                continue

        timestampstr = fn.split("_")[-1]
        if date is None:
            files.append(fn)
        elif timestampstr[:8] <= date:
            files.append(fn)
    try:
        return os.path.join(path, max(files))
    except ValueError:
        return None
    except Exception:
        raise


def read_holding(trade_date, strategy_path, offset=0, return_type="numpy", report_path="report"):
    """
    读持仓数据
    :param trade_date:
    :param strategy_path:
    :param offset: 持仓日期的偏移，offset=1为昨日
    :param return_type: numpy or pandas
    :param report_path: "report", 如果report下有多个目录，那么就是"report/optim_v0"
    :return:
    """
    date = __td__.get_previous_trading_date(trade_date, offset)
    if os.path.exists(os.path.join(strategy_path, "holding")):
        holding_path = os.path.join(strategy_path, "holding", "%s.csv" % date)
    else:
        holding_folder = os.path.join(strategy_path, report_path)
        # 找到最新的holding文件
        holding_path = get_last_file(holding_folder, key="holdings")

    # 为了确保中文路径也可以读
    with open(holding_path, "r") as infile:
        holding_df = pd.read_csv(infile, dtype={'trade_date': str, 'wind_code': str})
        holding_df = holding_df[holding_df.trade_date == date]

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
