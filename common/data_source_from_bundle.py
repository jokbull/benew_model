from common.configure import read_configure


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
