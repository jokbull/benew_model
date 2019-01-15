from collider.data.sensor import Sensor
from collider.data.message_package import MessagePackage
from scipy.stats import spearmanr
import numpy as np


class FakeForwardReturn(Sensor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lastvalue = None

    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp: MessagePackage, **kwargs):
        scaler = kwargs.get("scaler", 0.5)
        sigma = kwargs.get("sigma", 0.1)
        shrinkage = kwargs.get("shrinkage", 0.2)

        trueForwardReturn = mp.exposure
        fakeForwardReturn = trueForwardReturn * scaler + np.random.normal(scale=sigma, size=4000)

        if self.lastvalue is None:
            thisvalue = fakeForwardReturn

        else:
            thisvalue = self.lastvalue * (1 - shrinkage) + fakeForwardReturn * shrinkage

        self.lastvalue = thisvalue

        self.logger.debug(spearmanr(trueForwardReturn, thisvalue, nan_policy="omit")[0])

        return thisvalue, np.array(["fakeForwardReturn"])
