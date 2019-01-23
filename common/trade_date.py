import numpy as np


def generate_trade_date_sequence(trade_date):
    """
    生成交易日期序列
    :param trade_date: 如果是tuple, (start_date, end_date), 按开头结尾生成序列; 如果是
    :return: numpy.ndarray(dtype=str)
    """

    if isinstance(trade_date, tuple):
        from common.data_source_from_bundle import __td__
        result = __td__.get_trading_dates(trade_date[0], trade_date[1], inclusive=(True, True))
    elif isinstance(trade_date, str):
        result = np.array([trade_date])
    elif isinstance(trade_date, list):
        result = np.array(trade_date)
    elif isinstance(trade_date, np.ndarray):
        result = trade_date
    else:
        raise TypeError("not support trade_date type %s" % trade_date.__class__.__name__)
    return result
