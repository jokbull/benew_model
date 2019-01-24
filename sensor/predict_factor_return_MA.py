# -*- coding: utf-8 -*-
# @author: scrat

import numpy as np
from collider.data.sensor import Sensor

class PredictFactorReturn_MA(Sensor):
    """
    用MA模型对FactorReturn进行预测
    """

    @property
    def output_variables(self):
        return ["factorReturn", "factorName"]

    def do(self, date, mp, **kwargs):

        prediction_return = mp.factorReturn.mean(axis=0)
        for i, fac in enumerate(mp.factorName):
            rolling_window = kwargs.get(fac, 200)  # 默认使用200天的MA

            if mp.factorReturn.shape[0] >= rolling_window:
                prediction_return[i] = mp.factorReturn[(-rolling_window):, i].mean()
            else:
                # 当数量不足时,使用默认值, 即所有数据的mean
                self.logger.warning("factor %s return history is not enough?" % fac)

        return prediction_return, mp.factorName
