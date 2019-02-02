#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @license : Copyright(C), benew

# 2019-01-25. v1版本确定. @scrat


from collider.core.strategy import Strategy
from collider.sensor import *
from collider.utils.logger import user_log
from collider.sensor.get_factor_list import FACTOR_STYLE, FactorList
from collider.data.pipeline.sensor_flow import SensorFlow
from collider.utils.data_process import DataProcessing
from collider.data.data_manager import DataManager

from sensor.get_pool_v2 import GetPool
from sensor.estimation_factor_return_linear_regression import EstimationFactorReturn
from sensor.get_factor_data_v2 import GetFactorData
from sensor.orthogonalization import Orthogonalization_Schmidt
from sensor.exposure_concatenate import Concatenate


class flow_attribution(Strategy):
    def initialize(self):

        # 初始化各个sensor

        self.user_context.update("DM", DataManager(name=self.name))

        # 初始化辅助类

        self.user_context.update("alphaFactorDataFrame", FactorList(
            file=self.user_context.alpha_file,
            factor_style=FACTOR_STYLE.ALPHA
        ))

        self.user_context.update("riskFactorDataFrame", FactorList(
            file=self.user_context.risk_file,
            factor_style=FACTOR_STYLE.RISK
        ))

        # 初始化各个mod
        init_succeed = super().initialize()

        if init_succeed:
            self._init_estimation_flow()

        return init_succeed

    # region API
    def _init_estimation_flow(self):
        flow_name = self.user_context.flow_config.get("est_flow_name", "est_flow")
        self._estimation_flow = SensorFlow(name=flow_name, data_manager=self.user_context.DM)

        # factor date =  forward_period + 2
        self._estimation_flow.add_next_step2(name="factor_as_of_date",
                                             sensor=GetDate,
                                             kwds={'offset': self.user_context.forward_period + 2}
                                             )

        self._estimation_flow.add_next_step2(name="yesterday",
                                             sensor=GetDate,
                                             kwds={'offset': 1}
                                             )

        # module 4. 确定对Risk数据进行数据清洗的集合
        self._estimation_flow.add_next_step2(name="riskPool",
                                             sensor=GetPool,
                                             call=None,
                                             input_var=[f"{flow_name}.yesterday.date"],
                                             kwds={"pool_name": self.user_context.pool_name,
                                                   "benchmark_weight": "weight_index_500"
                                                   },
                                             silent=False)

        factorList = {}
        for k in self.user_context.riskFactorDataFrame.factor_dataFrame.factor:
            factorList[k] = FACTOR_STYLE.RISK

        # module 6. 取risk数据
        self._estimation_flow.add_next_step2(name="riskFactorData",
                                             sensor=GetFactorData,
                                             call=None,
                                             input_var=[f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.factor_as_of_date.date"
                                                        ],
                                             kwds={"data_process_methods": {
                                                 FACTOR_STYLE.SECTOR: [],
                                                 FACTOR_STYLE.RISK: [
                                                     DataProcessing.do_process_extremum_winsorize,
                                                     DataProcessing.do_z_score_processing
                                                 ]},
                                                 "factorList": factorList
                                             }
                                             )

        factorList = {}
        for k in self.user_context.alphaFactorDataFrame.factor_dataFrame.factor:
            factorList[k] = FACTOR_STYLE.ALPHA
        # module 7. 取alpha数据并正交
        self._estimation_flow.add_next_step2(name="alphaFactorData",
                                             sensor=GetFactorData,
                                             call=None,
                                             input_var=[f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.factor_as_of_date.date",
                                                        f"{flow_name}.riskFactorData.exposure"
                                                        ],
                                             kwds={"data_process_methods": {
                                                 FACTOR_STYLE.SECTOR: [],
                                                 FACTOR_STYLE.RISK: [
                                                     DataProcessing.do_process_extremum_winsorize,
                                                     DataProcessing.neutrialize,
                                                     DataProcessing.do_z_score_processing
                                                 ]},
                                                 "factorList": factorList
                                             }
                                             )

        self._estimation_flow.add_next_step2(name="orthogonalization",
                                             sensor=Orthogonalization_Schmidt,
                                             call=None,
                                             input_var=[f"{flow_name}.alphaFactorData.exposure",
                                                        f"{flow_name}.alphaFactorData.factorName",
                                                        f"{flow_name}.riskPool.pool"
                                                        ],
                                             kwds={})

        # module 8. concatenate
        self._estimation_flow.add_next_step2(name="riskAlphaFactorData",
                                             sensor=Concatenate,
                                             call=None,
                                             input_var=[f"{flow_name}.orthogonalization.exposure",
                                                        f"{flow_name}.orthogonalization.factorName",
                                                        f"{flow_name}.riskFactorData.exposure",
                                                        f"{flow_name}.riskFactorData.factorName",
                                                        ],
                                             alias={
                                                 f"{flow_name}.orthogonalization.exposure": "exposure_1",
                                                 f"{flow_name}.orthogonalization.factorName": "factorName_1",
                                                 f"{flow_name}.riskFactorData.exposure": "exposure_2",
                                                 f"{flow_name}.riskFactorData.factorName": "factorName_2"
                                             },
                                             kwds={}, silent=False)

        try:
            open_price_type = self.user_context.est_open_price
        except Exception as e:
            user_log.warning("no est_open_price in config file")
            open_price_type = "open_aft"

        try:
            close_price_type = self.user_context.est_close_price
        except Exception as e:
            user_log.warning("no est_close_price in config file")
            close_price_type = "open_aft"

        user_log.info("est open_price_type is : " + open_price_type)
        user_log.info("est close_price_type is : " + close_price_type)

        # module 8. 取return数据
        self._estimation_flow.add_next_step2(name="returnData",
                                             sensor=GetReturnData, call=None,
                                             input_var=[f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.factor_as_of_date.date"
                                                        ],
                                             kwds={"data_process_methods": [
                                                 # DataProcessing.do_process_extremum_winsorize,
                                                 # DataProcessing.neutrialize
                                             ],
                                                 "n": self.user_context.forward_period,
                                                 "open_price_type": open_price_type,
                                                 "close_price_type": close_price_type
                                             },
                                             silent=False
                                             )

        # module 10. 模型估计
        self._estimation_flow.add_next_step2(name="OLS",
                                             sensor=EstimationFactorReturn,
                                             call=None,
                                             input_var=[f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.returnData.stockReturn",
                                                        f"{flow_name}.riskAlphaFactorData.exposure",
                                                        f"{flow_name}.riskAlphaFactorData.factorName"],
                                             alias={
                                                 f"{flow_name}.riskAlphaFactorData.factorName": "name",
                                             },
                                             kwds={},
                                             silent=False
                                             )

    def before_trading(self, event):
        """
        默认在这里显式调用各个flow。
        """
        # 交易日，各个flow的输入日期都是交易日
        trade_date = event.trading_dt.strftime("%Y%m%d")

        user_log.info(f"{trade_date}")

        # fit
        self._estimation_flow.run(trade_date)

    def after_trading(self, event):
        self.user_context.DM.flush(event.trading_dt.strftime("%Y%m%d"))
