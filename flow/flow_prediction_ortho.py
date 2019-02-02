#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2018-07-06 Naive Prediction v1确定.  @scrat
# 2018-12-08 Naive Prediction v2. @scrat 简化


import numpy as np

from collider.core.strategy import Strategy
from collider.sensor import *
from collider.data.data_manager import DataManager
from collider.utils.logger import user_log
from collider.sensor.get_factor_list import FactorList, FACTOR_STYLE
from collider.data.pipeline.sensor_flow import SensorFlow
from collider.sensor.store.Store import Store

from sensor.get_factor_data_v2 import GetFactorData
from sensor.save_to_npy import SaveToBundleSensor


class flow_prediction_ortho(Strategy):

    # region API

    def initialize(self):
        # 初始化各个sensor

        self.user_context.update("DM", DataManager(name=self.name))

        # 初始化辅助类

        self.user_context.update("alphaFactorDataFrame", FactorList(
            file=self.user_context.alpha_file,
            factor_style=FACTOR_STYLE.ALPHA
        ))

        # 初始化各个mod
        init_succeed = super().initialize()

        if init_succeed:
            self._init_prepare_flow()
            self._init_store_flow()
            self._init_prediction_stock()

            self._prepare_flow.run(date="99991231")

        return init_succeed

    def _init_prepare_flow(self):
        self._prepare_flow = SensorFlow(name="prepare_flow", data_manager=self.user_context.DM)

        # module 1. 确定Alpha因子清单
        self._prepare_flow.add_next_step2(name="alphaFactorList",
                                          sensor=GetFactorList,
                                          kwds={"factor_list": self.user_context.alphaFactorDataFrame}
                                          )

    def _init_store_flow(self):
        est_flow = self.user_context.flow_config.get("est_flow_name", "estimation_flow")

        self._store_flow = SensorFlow(name="store_flow", data_manager=self.user_context.DM)

        self._store_flow.add_next_step2(name="store",
                                        sensor=Store, call=None,
                                        input_var=[
                                            "store_flow.store.factorReturn",
                                            "store_flow.store.factorName",
                                            "store_flow.store.dateList",
                                            f"{est_flow}.OLS.coefficient",
                                            "prepare_flow.alphaFactorList.factorList"
                                        ],
                                        alias={
                                            "store_flow.store.factorName": "col_names",
                                            "prepare_flow.alphaFactorList.factorList": "factorName"
                                        },
                                        kwds={}
                                        )

    def _init_prediction_stock(self):
        est_flow = self.user_context.flow_config.get("est_flow_name", "estimation_flow")
        pred_flow = self.user_context.flow_config.get("pred_flow_name", "prediction_stock")

        # silent = (not self.user_context.debug) if hasattr(self.user_context, "debug") else True

        self._prediction_stock = SensorFlow(pred_flow, data_manager=self.user_context.DM)

        # module 2. 预测factor_return
        self._prediction_stock.add_next_step2(name="predictionFactorReturn",
                                              sensor=PredictionFactorReturn,
                                              input_var=["store_flow.store.factorReturn",
                                                         "store_flow.store.factorName"],
                                              kwds={"rolling_window": self.user_context.rolling_window},
                                              silent=False
                                              )

        # module 3. 预测因子协方差矩阵
        self._prediction_stock.add_next_step2(name="predictionFactorCovariance",
                                              sensor=PredictionFactorCovariance,
                                              input_var=["store_flow.store.factorReturn",
                                                         "store_flow.store.factorName"],
                                              kwds={"rolling_window": self.user_context.rolling_window},
                                              silent=False
                                              )

        # 取昨日因子数据
        self._prediction_stock.add_next_step2(name="factor_as_of_date",
                                              sensor=GetDate,
                                              kwds={'offset': 1}
                                              )

        factorList = {}
        for k in self.user_context.alphaFactorDataFrame.factor_dataFrame.factor:
            factorList[k + "_f1"] = FACTOR_STYLE.SECTOR

        # module 7. 取alpha数据
        self._prediction_stock.add_next_step2(name="orthogonalization",
                                             sensor=GetFactorData,
                                             call=None,
                                             input_var=[
                                                 f"{pred_flow}.factor_as_of_date.date"
                                             ],
                                             kwds={"factorList": factorList})

        # module 7. 预测股票收益
        self._prediction_stock.add_next_step2(name="predictionStockReturn",
                                              sensor=PredictionStockReturn,
                                              input_var=[f"{pred_flow}.predictionFactorReturn.factorReturn",
                                                         f"{pred_flow}.orthogonalization.exposure",
                                                         f"{est_flow}.OLS.model"],
                                              kwds={},
                                              silent=False
                                              )

        self._prediction_stock.add_next_step2(
            name="saveToNpy_return",
            sensor=SaveToBundleSensor,
            call=None,
            input_var=[f"{pred_flow}.factor_as_of_date.date",
                       f"{pred_flow}.predictionStockReturn.stockReturn"],
            kwds={
                'bundle': self.user_context.config.base.data_bundle_path,
                'suffix': 'r1',
                'type': "return",
                'name': "predicted_stock_return"
            }
        )

    def before_trading(self, event):
        """
        默认在这里显式调用各个flow。
        """

        # 交易日，各个flow的输入日期都是交易日
        trade_date = event.trading_dt.strftime("%Y%m%d")

        user_log.info(f"{trade_date}")

        # 如果store里面数量为None，那么就是要preheat，把store_flow里面充满
        # 那么就preheat，silent=True
        store_num = self.user_context.DM.get_tensor(
            self.data_source.trading_dates.get_previous_trading_date(trade_date), "store_flow.store.factorReturn").data
        if store_num is None:
            for _date in self.data_source.trading_dates.get_previous_trading_date(
                    trade_date, self.user_context.rolling_window - np.arange(self.user_context.rolling_window)):
                self.user_context.DM.load_tensor(_date)
                self._store_flow.run(_date)

        # 当预测时，假定是每天开盘前进行预测

        self.user_context.DM.load_tensor(trade_date)
        self._store_flow.run(date=trade_date)
        self._prediction_stock.run(date=trade_date)

    def after_trading(self, event):
        self.user_context.DM.flush(event.trading_dt.strftime("%Y%m%d"))

    # FIXME: 需要显式调用Sensor的析构函数, 可能和Sensor是Singleton有关(永远不会自动触发）
    def end(self, event):
        for k in self._prediction_stock._pipeline:
            try:
                k._sensor.__del__()
            except AttributeError:
                pass

