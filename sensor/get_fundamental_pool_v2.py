# -*- coding: utf-8 -*-
# @author: scrat

import numpy as np

from collider.data.data_manager import DataManager
from collider.data.message_package import MessagePackage
from collider.data.sensor import Sensor

class GetFundamentalPool(Sensor):
    @property
    def output_variables(self):
        return ['pool']

    def do(self, date: str, mp: MessagePackage, **kwargs) -> tuple:

        as_of_date = mp.date
        next_date = mp.data_manager.trading_dates.get_next_trading_date(as_of_date)


        pool_name = kwargs["pool_name"]
        quantile = kwargs.get("threshold", 0.5)

        if quantile < 0 or quantile > 1:
            raise ValueError("threshold should belong to [0, 1]")

        pool_codes_dict = mp.data_manager.get_bar(
            date=as_of_date,
            columns=[pool_name, "pool_01"]
        )
        zscore = pool_codes_dict[pool_name]

        pool_index = zscore >= np.nanpercentile(zscore, q=int(quantile * 100))

        # 如果有基准，那么把基准的成分都加入pool
        if "benchmark_weight" in kwargs:
            benchmark = kwargs["benchmark_weight"]
            benchmark_weight = mp.data_manager.get_bar(date=as_of_date, columns=[benchmark])[benchmark]
            pool_index |= benchmark_weight > 0

        # 如果存在当前持仓
        if hasattr(mp, "weight"):
            pool_index |= mp.weight > 0

        # 去掉st股票
        st = mp.data_manager.get_bar(date=next_date, columns=["is_st"])["is_st"]
        is_st = st == 1
        pool_index &= (~is_st)

        pool_index &= pool_codes_dict["pool_01"] == 1
        return pool_index,
