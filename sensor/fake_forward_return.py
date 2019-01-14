from collider.data.sensor import Sensor
from collider.data.message_package import MessagePackage
from scipy.stats import spearmanr
import numpy as np

class FakeForwardReturn(Sensor):


    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp: MessagePackage, **kwargs):

        trueForwardReturn = mp.exposure
        fakeForwardReturn = trueForwardReturn * 0.2 + np.random.normal(scale=0.05, size=4000)


        print(spearmanr(trueForwardReturn, fakeForwardReturn, nan_policy="omit")[0])

        return fakeForwardReturn, np.array(["fakeForwardReturn"])