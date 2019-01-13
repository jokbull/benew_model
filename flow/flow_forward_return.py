#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @license : Copyright(C), benew

# 2018-07-06. v1版本确定. @scrat
# 2018-12-08. v2版本: est/pred残差一致. @scrat


from collider.core.strategy import Strategy
from collider.sensor import *
from collider.data.data_manager import DataManager
from collider.utils.logger import user_log
from collider.utils.data_process import DataProcessing
from collider.sensor.get_factor_list import FactorList, FACTOR_STYLE
from collider.data.pipeline.sensor_flow import SensorFlow

from sensor.get_pool_v2 import GetPool
from sensor.save_to_npy import SaveToBundleSensor
from sensor.get_factor_data_v2 import GetFactorData


class flow_forward_return(Strategy):

    # region API

    def initialize(self):

        # 初始化各个sensor

        self.user_context.update("DM", DataManager(name=self.name))

        # 初始化辅助类

        self.user_context.update("riskFactorDataFrame", FactorList(
            file=self.user_context.risk_file,
            factor_style=FACTOR_STYLE.RISK
        ))

        # 初始化各个mod
        init_succeed = super().initialize()

        if init_succeed:
            self._init_estimation_flow()

        return init_succeed

    def _init_estimation_flow(self):
        flow_name = self.user_context.flow_config.get("est_flow_name", "est_flow")
        self._estimation_flow = SensorFlow(name=flow_name, data_manager=self.user_context.DM)

        # factor date
        self._estimation_flow.add_next_step2(name="factor_as_of_date",
                                             sensor=GetDate,
                                             kwds={'offset': self.user_context.forward_period + 2}
                                             )

        # module 4. 确定对Risk数据进行数据清洗的集合
        self._estimation_flow.add_next_step2(name="riskPool",
                                             sensor=GetPool,
                                             call=None,
                                             # 这里用factor_as_of_date
                                             input_var=[f"{flow_name}.factor_as_of_date.date"],
                                             kwds={"pool_name": self.user_context.pool_name})

        factorList = {}
        for k in self.user_context.riskFactorDataFrame.factor_dataFrame.factor:
            factorList[k] = FACTOR_STYLE.SECTOR if k.startswith("industry") else FACTOR_STYLE.RISK

        # module 6. 取risk数据
        self._estimation_flow.add_next_step2(name="riskFactorData",
                                             sensor=GetFactorData,
                                             call=None,
                                             input_var=[f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.factor_as_of_date.date"
                                                        ],
                                             kwds={"factorList": factorList,
                                                   "data_process_methods": {
                                                       FACTOR_STYLE.SECTOR: [],
                                                       FACTOR_STYLE.RISK: [
                                                           DataProcessing.do_process_extremum_winsorize,
                                                           DataProcessing.do_z_score_processing
                                                       ]
                                                   }},
                                             silent=False)

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
                                             input_var=[f"{flow_name}.riskFactorData.exposure",
                                                        f"{flow_name}.riskPool.pool",
                                                        f"{flow_name}.factor_as_of_date.date"
                                                        ],
                                             alias={
                                                 f"{flow_name}.riskFactorData.exposure": "neutralize_matrix",
                                             },
                                             kwds={"data_process_methods": [
                                                 DataProcessing.do_process_extremum_winsorize,
                                                 DataProcessing.neutrialize],
                                                 "n": self.user_context.forward_period,
                                                 "open_price_type": open_price_type,
                                                 "close_price_type": close_price_type
                                             },
                                             silent=True)

        self._estimation_flow.add_next_step2(
            name="saveToNpy_return",
            sensor=SaveToBundleSensor,
            call=None,
            input_var=[f"{flow_name}.factor_as_of_date.date",
                       f"{flow_name}.returnData.stockReturn"],
            kwds={
                'bundle': self.user_context.config.base.data_bundle_path,
                'suffix': 'f1',
                'type': "return",
                'name': "forward_return_%d" % self.user_context.forward_period
            }
        )

    def before_trading(self, event):
        self.user_context.DM.load_tensor(event.trading_dt.strftime("%Y%m%d"))
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

    # FIXME: 需要显式调用Sensor的析构函数, 可能和Sensor是Singleton有关(永远不会自动触发）
    def end(self, event):
        for k in self._estimation_flow._pipeline:
            try:
                k._sensor.__del__()
            except AttributeError:
                pass
