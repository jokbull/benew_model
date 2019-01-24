# -*- coding: utf-8 -*-
# @author: scrat

import numpy as np
from collider.data.sensor import Sensor
from pmdarima.arima import auto_arima


class PredictFactorReturn_ARIMA(Sensor):
    """
    用ARIMA模型对FactorReturn进行预测
    """

    @property
    def output_variables(self):
        return ["factorReturn", "factorName", "arima"]

    def do(self, date, mp, **kwargs):

        rolling_window = kwargs.setdefault("rolling_window", 1)
        arima_dict = {}

        if mp.factorReturn.shape[0] >= rolling_window:
            prediction_return = np.zeros(shape=len(mp.factorName))
            for i, fac in enumerate(mp.factorName):
                train = mp.factorReturn[(-rolling_window), i]
                arima = auto_arima(train,
                                   start_p=1, start_q=1, d=0,
                                   max_p=5, max_d=2, max_q=5,
                                   seasonal=False)
                arima_dict[fac] = arima
                prediction_return[i] = arima.predict(n_periods=kwargs.get("forward_period", 5))[-1]
        else:
            # 当数量不足时,返回nan
            self.logger.warning("factor return history is not enough?")
            prediction_return = np.full(mp.factorName.size, np.nan)

        return prediction_return, mp.factorName, arima_dict
