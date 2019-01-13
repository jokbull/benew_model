# -*- coding: utf-8 -*-
# @author: scrat


import numpy as np
from collider.data.sensor import Sensor
from collider.utils.data_process import DataProcessingWithMask

"""
"""


class Orthogonalization_Rotating(Sensor):

    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp, **kwargs):

        column = mp.factorName
        data = mp.exposure
        mask = mp.pool

        if len(data.shape) == 1:
            return data, np.array(column)
        else:
            dropna = np.isnan(data).any(axis=1)
        if all(dropna):
            self.logger.warn("X are all NA")
            return None, None, None

        # drop na
        F = data[mask & ~dropna]

        # Sigma
        Sigma = np.cov(F.T)
        # M
        M = (len(F) - 1) * Sigma
        # U, D
        temp, U = np.linalg.eig(M)
        D = np.diag(1 / temp)
        # S
        S = np.dot(np.dot(U, np.sqrt(D)), U.T)
        # orth F
        orth_F = np.dot(data, S)
        scaler = np.diag(Sigma) / np.diag(np.cov(orth_F[mask & ~dropna].T))
        orth_scale_F = np.dot(orth_F, np.diag(np.sqrt(scaler)))

        return orth_F, np.array(column)


class Orthogonalization_Schmidt(Sensor):

    @property
    def output_variables(self):
        return ["exposure", "factorName"]

    def do(self, date, mp, **kwargs):

        column = mp.factorName
        data = mp.exposure
        mask = mp.pool

        if len(data.shape) == 1:
            return data, np.array(column)
        else:
            dropna = np.isnan(data).any(axis=1)
        if all(dropna):
            self.logger.warn("X are all NA")
            return None, None, None

        m = mask & ~dropna

        result = None
        for i in range(data.shape[1]):
            if i == 0:
                result = np.array([data[:, i]]).T
            else:
                try:
                    thiscol = DataProcessingWithMask.neutrialize(
                        array=data[:, i],
                        mask=m,
                        neutralize_matrixX=result,
                        hasconst=True)  # 必须加常数项，保证正交
                except Exception as e:
                    raise (e)
                result = np.c_[result, thiscol]

        return result, np.array(column)
