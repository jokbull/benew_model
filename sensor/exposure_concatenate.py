from collider.data.message_package import MessagePackage
from collider.data.sensor import Sensor
import numpy as np


class Concatenate(Sensor):

    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date: str, mp: MessagePackage, **kwargs) -> tuple:

        exposure_1 = mp.exposure_1
        name_1 = mp.factorName_1
        exposure_2 = mp.exposure_2
        name_2 = mp.factorName_2

        try:
            exposure = np.c_[exposure_1, exposure_2]
            name = np.concatenate([name_1, name_2])
            return exposure, name
        except Exception:
            raise
