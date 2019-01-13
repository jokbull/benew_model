# -*- coding: utf-8 -*-
# @author: scrat


import statsmodels.api as sm
import numpy as np
from collider.data.sensor import Sensor


class EstimationFactorReturn(Sensor):

    @property
    def output_variables(self):
        return ["date", "pool", "factor_name", "model", "modelResult", "coefficient", "residual", "fitted"]

    def do(self, date, mp, **kwargs):
        trading_dt = date
        # 如果是WLS, 那么weight往往也是要用一个Sensor来load的.
        weight = mp.weight if hasattr(mp, "weight") else None

        # 准备数据
        x_name = mp.name
        mask = mp.pool
        x = mp.exposure
        y = mp.stockReturn

        # https://stackoverflow.com/questions/11453141/how-to-remove-all-rows-in-a-numpy-ndarray-that-contain-non-numeric-values
        if len(x.shape) == 1:
            dropna = np.isnan(x)
        else:
            dropna = np.isnan(x).any(axis=1)
        if all(dropna):
            self.logger.warn("X are all NA")
            return None, None, None

        # constant
        hasconst = kwargs["hasconst"] if "hasconst" in kwargs else False
        if hasconst:
            x = sm.add_constant(x)

        # fit dataset
        x_adj = x[(~dropna) & mask]
        y_adj = y[(~dropna) & mask]

        if weight:
            model = sm.WLS(y_adj, x_adj, hasconst=False, weights=weight, missing="drop")
        else:
            model = sm.OLS(y_adj, x_adj, hasconst=False, missing="drop")

        result = model.fit()
        coefficients = result.params
        # residual in the whole data set
        if len(coefficients) == 1:
            fitted = model.predict(result.params, np.array([x]).T)
        else:
            fitted = model.predict(result.params, x)
        # print(1 - np.nanmean(residuals ** 2) / np.nanvar(y))

        residuals = y - fitted
        return trading_dt, (~dropna) & mask, x_name, model, result, coefficients, residuals, fitted
