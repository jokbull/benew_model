from collider.data.sensor import Sensor
from collider.data.message_package import MessagePackage
from scipy.stats import spearmanr
import numpy as np


class FakeForwardReturn(Sensor):

    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp: MessagePackage, **kwargs):
        scaler = kwargs.get("scaler", 0.1)
        sigma = kwargs.get("sigma", 0.02)

        trueForwardReturn = mp.exposure
        fakeForwardReturn = trueForwardReturn * scaler + np.random.normal(scale=sigma, size=4000)

        self.logger.debug(spearmanr(trueForwardReturn, fakeForwardReturn, nan_policy="omit")[0])

        return fakeForwardReturn, np.array(["fakeForwardReturn"])
