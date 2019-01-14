from model.breakdown.Optimization.L1Norm.L1Norm import *
from collider.sensor.optimization_stock_weight_v2 import OptimizationStockWeightV2
from collider.utils.logger import user_log
from collider.const import FACTOR_STYLE
from collider.sensor.get_factor_list import FactorList
from sensor.get_factor_data_v2 import GetFactorData
from sensor.fake_forward_return import FakeForwardReturn
from sensor.save_to_npy import SaveToBundleSensor


class flow_fake_forward_return(Strategy):
    def initialize(self):

        # 初始化各个sensor

        self.user_context.update("DM", DataManager(name=self.name))

        # 初始化辅助类
        # 初始化各个mod
        init_succeed = super().initialize()

        if init_succeed:
            self._init_optim()

        return init_succeed



    def _init_optim(self):
        optim_flow = self.user_context.flow_config.get("flow_name", "fake_forward_return")

        self._optim_flow = SensorFlow(name=optim_flow, data_manager=self.user_context.DM)

        # module 19. 昨日持仓
        self._optim_flow.add_next_step(sensor=GetDate,
                                       args=["factor_as_of_date", [], {}],
                                       kwds={'offset': 1}
                                       )

        # module 7. 取true_forward_return数据
        self._optim_flow.add_next_step2(name="forwardReturnData",
                                        sensor=GetFactorData,
                                        call=None,
                                        input_var=[
                                            f"{optim_flow}.factor_as_of_date.date"
                                        ],
                                        kwds={"factorList": {'forward_return_5_f1': FACTOR_STYLE.ALPHA}})

        # 对true_forward_return加入噪声
        fake_settings = {
            'scaler': 0.1,
            'sigma': 0.02
        }
        self._optim_flow.add_next_step2(name="fakeForwardReturnData",
                                        sensor=FakeForwardReturn, call=None,
                                        input_var=[
                                            f"{optim_flow}.forwardReturnData.exposure"
                                        ],
                                        kwds=fake_settings)

        self._optim_flow.add_next_step2(
            name="saveToNpy_return",
            sensor=SaveToBundleSensor,
            call=None,
            input_var=[f"{optim_flow}.factor_as_of_date.date",
                       f"{optim_flow}.fakeForwardReturnData.exposure"],
            alias={
                f"{optim_flow}.fakeForwardReturnData.exposure": "stockReturn"
            },
            kwds={
                'bundle': self.user_context.config.base.data_bundle_path,
                'suffix': 'f1',
                'type': "return",
                'name': optim_flow
            }
        )

        # module 8. 取fitted_forward_return(也是用到未来数据）
        self._optim_flow.add_next_step2(name="fittedForwardReturnData",
                                        sensor=GetFactorData,
                                        call=None,
                                        input_var=[
                                            f"{optim_flow}.factor_as_of_date.date"
                                        ],
                                        kwds={"factorList": {'flow_estimation_fitted_f1': FACTOR_STYLE.ALPHA}})

    def before_trading(self, event):
        """
        默认在这里显式调用各个flow。
        """
        # 交易日，各个flow的输入日期都是交易日
        trade_date = event.trading_dt.strftime("%Y%m%d")

        user_log.info(f"{trade_date}")

        # fit
        self._optim_flow.run(trade_date)

    # FIXME: 需要显式调用Sensor的析构函数, 可能和Sensor是Singleton有关(永远不会自动触发）
    def end(self, event):
        for k in self._optim_flow._pipeline:
            try:
                k._sensor.__del__()
            except AttributeError:
                pass