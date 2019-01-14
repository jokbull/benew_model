from model.breakdown.Optimization.L1Norm.L1Norm import *
from collider.const import FACTOR_STYLE
from collider.sensor.get_factor_list import FactorList
from collider.utils.data_process import DataProcessing

from sensor.optimization_stock_weight_v2tc import OptimizationStockWeightV2tc
from sensor.get_fundamental_pool_v2 import GetFundamentalPool
from sensor.get_factor_data_v2 import GetFactorData
from sensor.fake_forward_return import FakeForwardReturn


class flow_optim_forward_return(L1Norm):
    def initialize(self):
        # self.user_context.update("pool_name", "pool_01")

        self.user_context.update("alphaFactorDataFrame", FactorList(
            file=self.user_context.alpha_file,
            factor_style=FACTOR_STYLE.ALPHA
        ))

        self.optim_options = {"options": {"show_progress": False},
                              "tc": 0.003,
                              "top": 500,
                              "risk_condition": {
                                  "up": {
                                      'style_size_2': 0.0005,
                                      'style_beta_2': 0.0005,
                                      'industry_商贸零售': 0.03,
                                      'industry_石油石化': 0.03,
                                      'industry_国防军工': 0.03,
                                      'industry_传媒': 0.03,
                                      'industry_餐饮旅游': 0.03,
                                      'industry_汽车': 0.03,
                                      'industry_电力及公用事业': 0.03,
                                      'industry_电力设备': 0.03,
                                      'industry_综合': 0.03,
                                      'industry_计算机': 0.03,
                                      'industry_医药': 0.03,
                                      'industry_建材': 0.03,
                                      'industry_农林牧渔': 0.03,
                                      'industry_机械': 0.03,
                                      'industry_纺织服装': 0.03,
                                      'industry_保险Ⅱ': 0.03,
                                      'industry_食品饮料': 0.03,
                                      'industry_信托及其他': 0.03,
                                      'industry_电子元器件': 0.03,
                                      'industry_煤炭': 0.03,
                                      'industry_建筑': 0.03,
                                      'industry_银行': 0.03,
                                      'industry_基础化工': 0.03,
                                      'industry_证券Ⅱ': 0.03,
                                      'industry_家电': 0.03,
                                      'industry_交通运输': 0.03,
                                      'industry_钢铁': 0.03,
                                      'industry_有色金属': 0.03,
                                      'industry_通信': 0.03,
                                      'industry_轻工制造': 0.03,
                                      'industry_房地产': 0.03
                                  },
                                  "down": {
                                      'style_size_2': 0.0005,
                                      'style_beta_2': 0.0005,
                                      'industry_商贸零售': 0.03,
                                      'industry_石油石化': 0.03,
                                      'industry_国防军工': 0.03,
                                      'industry_传媒': 0.03,
                                      'industry_餐饮旅游': 0.03,
                                      'industry_汽车': 0.03,
                                      'industry_电力及公用事业': 0.03,
                                      'industry_电力设备': 0.03,
                                      'industry_综合': 0.03,
                                      'industry_计算机': 0.03,
                                      'industry_医药': 0.03,
                                      'industry_建材': 0.03,
                                      'industry_农林牧渔': 0.03,
                                      'industry_机械': 0.03,
                                      'industry_纺织服装': 0.03,
                                      'industry_保险Ⅱ': 0.03,
                                      'industry_食品饮料': 0.03,
                                      'industry_信托及其他': 0.03,
                                      'industry_电子元器件': 0.03,
                                      'industry_煤炭': 0.03,
                                      'industry_建筑': 0.03,
                                      'industry_银行': 0.03,
                                      'industry_基础化工': 0.03,
                                      'industry_证券Ⅱ': 0.03,
                                      'industry_家电': 0.03,
                                      'industry_交通运输': 0.03,
                                      'industry_钢铁': 0.03,
                                      'industry_有色金属': 0.03,
                                      'industry_通信': 0.03,
                                      'industry_轻工制造': 0.03,
                                      'industry_房地产': 0.03
                                  }
                              }}
        self.user_context.update("optim_options", {})

        return super().initialize()

    def _init_optim(self):
        prediction_flow = self.user_context.flow_config.get("pred_flow_name", "prediction_stock")
        optim_flow = self.user_context.flow_config.get("optim_flow_name", "optim_flow")
        forward_return_flow = self.user_context.flow_config.get("forward_return_flow", "flow_forward_return")

        self._optim_flow = SensorFlow(name=optim_flow, data_manager=self.user_context.DM)

        # module 19. 昨日持仓
        self._optim_flow.add_next_step(sensor=GetHolding,
                                       args=["holding", [], {}],
                                       kwds={"account": self.account})

        self._optim_flow.add_next_step(sensor=GetDate,
                                       args=["factor_as_of_date", [], {}],
                                       kwds={'offset': 1}
                                       )

        # module 11. 确定对Alpha/Risk数据进行数据清洗的集合
        self._optim_flow.add_next_step(sensor=GetFundamentalPool,
                                       args=["stockCandidate",
                                             [
                                                 f"{optim_flow}.holding.weight",
                                                 f"{optim_flow}.factor_as_of_date.date"
                                             ], {}],
                                       kwds={"pool_name": self.user_context.ff_name,
                                             "threshold": 0.3,
                                             # "benchmark_weight": "weight_index_500"
                                             },
                                       silent=False)

        factorList = {}
        for k in self.user_context.alphaFactorDataFrame.factor_dataFrame.factor:
            factorList[k + "_f1"] = FACTOR_STYLE.ALPHA

        # module 7. 取alpha数据
        self._optim_flow.add_next_step2(name="alphaPredData",
                                        sensor=GetFactorData,
                                        call=None,
                                        input_var=[
                                            f"{optim_flow}.factor_as_of_date.date"
                                        ],
                                        kwds={"factorList": factorList})

        # module 7. 取true_forward_return数据
        self._optim_flow.add_next_step2(name="forwardReturnData",
                                        sensor=GetFactorData,
                                        call=None,
                                        input_var=[
                                            f"{optim_flow}.factor_as_of_date.date"
                                        ],
                                        kwds={"factorList": {'forward_return_5_f1': FACTOR_STYLE.ALPHA}})

        # 对true_forward_return加入噪声
        self._optim_flow.add_next_step2(name="fakeForwardReturnData",
                                        sensor=FakeForwardReturn, call=None,
                                        input_var=[
                                            f"{optim_flow}.forwardReturnData.exposure"
                                        ],
                                        kwds={})


        # module 8. 取fitted_forward_return(也是用到未来数据）
        self._optim_flow.add_next_step2(name="fittedForwardReturnData",
                                        sensor=GetFactorData,
                                        call=None,
                                        input_var=[
                                            f"{optim_flow}.factor_as_of_date.date"
                                        ],
                                        kwds={"factorList": {'flow_estimation_fitted_f1': FACTOR_STYLE.ALPHA}})


        kwds = {}
        kwds.update(self.optim_options)
        kwds.update({'total_value': 10000000})
        # kwds.update({'tc_b': 5})
        # kwds.update({'tc_a': 0.5})
        self._optim_flow.add_next_step(sensor=OptimizationStockWeightV2tc,
                                       args=["optimizationStockWeight", [

                                           "%s.fittedForwardReturnData.exposure" % optim_flow,
                                           "%s.predictionFactorCovariance.factorCovariance" % prediction_flow,

                                           "%s.alphaPredData.exposure" % optim_flow,
                                           "%s.alphaPredData.factorName" % optim_flow,
                                           "%s.riskFactorData.exposure" % forward_return_flow,
                                           "%s.riskFactorData.factorName" % forward_return_flow,

                                           "%s.stockCandidate.pool" % optim_flow,
                                           "%s.holding.weight" % optim_flow,
                                           "%s.factor_as_of_date.date" % optim_flow
                                       ], {
                                                 "%s.alphaPredData.exposure" % optim_flow: "alphaExposure",
                                                 "%s.alphaPredData.factorName" % optim_flow: "alphaName",
                                                 "%s.riskFactorData.exposure" % forward_return_flow: "riskExposure",
                                                 "%s.riskFactorData.factorName" % forward_return_flow: "riskName",
                                                 "%s.fittedForwardReturnData.exposure" % optim_flow: 'stockReturn'
                                             }],
                                       kwds=kwds,
                                       silent=False)

        self._optim_flow.add_next_step2(name="dumpTargetWeight",
                                        sensor=DumpTargetWeight,
                                        kwds={},
                                        input_var=[f"{optim_flow}.optimizationStockWeight.targetWeight"]
                                        )
