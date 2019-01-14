# -*- coding: utf-8 -*-
# @author: scrat


import numpy as np
import os
from collider.data.message_package import MessagePackage
from collider.data.sensor import Sensor


class SaveToBundle(object):
    def __init__(self, **kwargs):
        if not 'factor_name' in kwargs:
            raise AttributeError("provide factor_name")

        factor_name = kwargs['factor_name']
        bundle_path = kwargs.get('bundle_path', '/data/bundle')
        self.filename = os.path.join(bundle_path, 'factor', f"{factor_name}.npy")
        if os.path.exists(self.filename):
            raise FileExistsError(self.filename)

        # codes = np.load(os.path.join(bundle_path, 'code.npy'))
        self.dates = np.load(os.path.join(bundle_path, 'date.npy'))
        shape = (len(self.dates) * 4000,)
        dtype = '<f8'
        self.fp = np.empty(shape=shape, dtype=dtype)
        # self.fp = np.memmap(self.filename, dtype=dtype, mode='w+', shape=shape)

    def save(self, trade_date, arr):
        assert (arr.shape == (4000,))

        date_index = np.searchsorted(self.dates, trade_date)
        if date_index == len(self.dates):
            raise KeyError("not found %s" % trade_date)
        self.fp[(date_index * 4000):(date_index * 4000 + 4000)] = arr

    def flush(self):
        # self.fp.flush()
        np.save(file=self.filename, arr=self.fp)

    def __del__(self):
        print("__del__ %s" % self.filename)
        self.flush()


class SaveToBundleSensor(Sensor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fp_dict = {}

    def do(self, date: str, mp: MessagePackage, **kwargs) -> tuple:

        datatype = kwargs.get("type", "factor")
        outpath = kwargs.get("bundle", "/data/bundle")
        suffix = kwargs.get("suffix", "")

        if datatype == "pool":
            data = mp.pool
            name = [kwargs.get("name")]
            data = data.reshape(data.shape[0], 1)
        elif datatype == "return":
            data = mp.stockReturn
            name = [kwargs.get("name")]
            data = data.reshape(data.shape[0], 1)
        elif datatype == "factor":
            data = mp.exposure
            name = mp.factorName

        else:
            raise AttributeError(f"Not support {datatype}")

        assert (data.shape[1] == len(name))
        for i, n in enumerate(name):
            if not n in self.fp_dict:
                self.fp_dict[n] = SaveToBundle(factor_name="%s_%s" % (n, suffix), bundle_path=outpath)

            self.fp_dict[n].save(trade_date=mp.date, arr=data[:, i])

        return ()

    def __del__(self):
        for k in self.fp_dict.keys():
            self.fp_dict[k].__del__()


class SaveToNPY(Sensor):

    def do(self, date: str, mp: MessagePackage, **kwargs) -> tuple:

        datatype = kwargs.get("type", "factor")
        outpath = kwargs.get("path", "./clean_data")
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if datatype == "pool":
            data = mp.pool
            name = [kwargs.get("name")]
            data = data.reshape(data.shape[0], 1)
        elif datatype == "return":
            data = mp.stockReturn
            name = [kwargs.get("name")]
            data = data.reshape(data.shape[0], 1)
        elif datatype == "factor":
            data = mp.exposure
            name = mp.factorName

        else:
            raise AttributeError(f"Not support {datatype}")

        assert (data.shape[1] == len(name))
        for i, n in enumerate(name):
            outfile = os.path.join(outpath, f"{n}_{mp.date}.npy")
            np.save(file=outfile, arr=data[:, i])

        return ()
