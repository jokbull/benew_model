# -*- coding: utf-8 -*-
# @author: scrat


import numpy as np
from collider.data.sensor import Sensor

"""
依赖于GetFactorList,
在这里定义Factor的处理方式,
默认是不处理
"""


class GetFactorData(Sensor):


    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp, **kwargs):

        data_process_methods = kwargs.get("data_process_methods", {})

        as_of_date = mp.date

        if hasattr(mp, "factorList"):
            factorList = mp.factorList
        else:
            factorList = kwargs.get("factorList", {})

        if hasattr(mp, "pool"):
            pool = mp.pool
        else:
            pool = np.full(4000, True)

        result = None
        column = []
        for key, value in factorList.items():
            x = mp.data_manager.get_bar(date=as_of_date, columns=[key])[key]

            if np.any((x != 0) & (np.isfinite(x))):
                try:
                    if hasattr(mp, "exposure"):
                        for f in data_process_methods.get(value, []):
                            x = f(x, mask=pool, neutralize_matrixX=mp.exposure, **kwargs)
                    else:
                        for f in data_process_methods.get(value, []):
                            x = f(x, mask=pool, **kwargs)
                    nonan_sum = np.sum(~np.isnan(x[pool]))
                    if nonan_sum < 500:
                        self.logger.warn("date {} ,{} miss too much after preprocessing,{} < 500".format(as_of_date,key,nonan_sum))

                    if result is None:
                        result = x
                    else:
                        result = np.c_[result, x]

                    column.append(key)
                except Exception as ex:
                    self.logger.warn('%s, date: %s, factor: %s' % (ex, as_of_date, key))

            else:
                # 因子值全是nan的情况，填充1（20160107）
                self.logger.warning("date ,{} ,factor {} all nan ,replace nan by 1 ".format(as_of_date, key))
                if result is None:
                    result = np.full(len(x),1)
                else:
                    result = np.c_[result, np.full(len(x),1)]
                column.append(key)

        return result, np.array(column)
