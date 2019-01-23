from common.configure import read_configure
from collider.utils.logger import system_log
import numpy as np

system_log.level_name = "INFO"

def data_source(bundle_path=None, **kwargs):
    if bundle_path is None:
        bundle_path = read_configure(**kwargs)["bundle_path"]

    from collider.data.base_data_source import BaseDataSource
    ds = BaseDataSource()
    ds.initialize(bundle_path)
    return ds


def trading_dates(bundle_path=None, **kwargs):
    if bundle_path is None:
        bundle_path = read_configure(**kwargs)["bundle_path"]

    import numpy as np
    import os.path
    from collider.utils.trading_dates import TradingDates

    return TradingDates(np.load(os.path.join(bundle_path, "date.npy")))


__ds__ = data_source()
__td__ = trading_dates()


def load_data_from_npy(trade_date, factor_name, return_type="numpy"):
    if isinstance(factor_name, str):
        factor_name = [factor_name]
    result = __ds__.get_bar(trade_date, factor_name)
    if return_type == "numpy":
        if len(factor_name) == 1:
            return result[factor_name[0]]
        else:
            return np.r_[[result[fac] for fac in factor_name]]
    elif return_type == "dict":
        return result