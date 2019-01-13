# -*- coding: utf-8 -*-


import numpy as np
from cvxopt import solvers, matrix
from collider.data.sensor import Sensor
from collider.utils.logger import user_log

import cvxpy as cp


class OptimizationStockWeightV2tc(Sensor):

    @property
    def output_variables(self):
        return ["targetWeight"]

    def do(self, date, mp, **kwargs):

        # region 读入参数, 对应输入的数据

        # 优化器的参数
        lambdax = kwargs.get("lambdax", 1)  # lambda = 0.5?
        tc_a = kwargs.get("tc_a", 0.5)  # 交易惩罚项中参数
        tc_b = kwargs.get("tc_b", 1)  # 交易惩罚项中参数
        tc_power = kwargs.get("tc_power", 1.5)  # 交易惩罚项中参数
        tc_c = kwargs.get("tc_power",0)
        n = kwargs.get("top", 200)  # 前n个股票进入优化器
        single_max = kwargs.get("single_max", 0.02)  # 个股最大权重
        total_value = kwargs.get("total_value", 1000000)

        # benchmark weight
        weight_index = kwargs.get("benchmark_weight", "weight_index_500")

        # 因子矩阵
        column = mp.alphaName
        exog = mp.alphaExposure

        # 行业风格矩阵

        risk_column = mp.riskName
        risk_factor = mp.riskExposure

        # 协方差矩阵
        cov = mp.factorCovariance

        # 特质风险
        if hasattr(mp, "sp_risk"):
            sp = mp.sp_risk
        else:
            sp = np.zeros_like(mp.stockReturn)

        # 停牌股票, non_suspend全是True/False, 没有nan
        is_suspend = kwargs.get("is_susp", np.full(mp.stockReturn.size, 0))
        non_suspend = is_suspend == 0

        # 计算benchmark因子暴露
        benchmark_exposure = mp.data_manager.get_bar(date=mp.date, columns=[weight_index])[weight_index]
        benchmark_exposure = np.nan_to_num(benchmark_exposure) / np.nansum(benchmark_exposure)
        benchmark_expo = np.dot(benchmark_exposure, np.nan_to_num(risk_factor))

        # endregion

        success = False
        while (not success) and n < 1500:
            stock_return = mp.stockReturn.copy()

            # region 计算进行优化的股票集合
            # 1. mp.pool中计算top_flag
            # 2. holding | top_flag
            # 3. 因子不缺

            # step 1. 在mp.pool中计算top_flag
            stock_return[~mp.pool] = np.nan  # 这里在while-loop中虽然是重复计算，但是为了代码的可读性，还是放在loop里面
            non_nan_cnt = np.sum(~np.isnan(stock_return))
            if non_nan_cnt < n:
                self.logger.warning("non_nan_cnt(%s) < n(%s)" % (non_nan_cnt, n))
                n = non_nan_cnt
            return_ordered_index = np.argsort(-stock_return)[:non_nan_cnt]
            top_flag = np.full(stock_return.size, False, dtype=bool)
            top_flag[return_ordered_index[:n]] = True

            candidates = top_flag.copy()  # 在top_flag中的肯定是有predicted_stock_return的，所以数据肯定不缺失
            # 在candidates中去掉其他nan的情况
            # case 1. special_risk为nan
            # candidates &= ~np.isnan(sp)

            # 在candidates中还需要加入以下的情况
            # case 1. 持仓 且 有stock_return (即数据不缺失）
            candidates |= (mp.weight > 0) & (~np.isnan(stock_return))

            # to solve : 待求解变量w
            w = cp.Variable(np.sum(candidates))

            # 持仓 且 停牌 且 数据缺失
            holding_suspend = (mp.weight > 0) & (is_suspend == 1) & (np.isnan(stock_return))
            holding_suspend_sum = np.sum(mp.weight[holding_suspend])
            candidates_cnt = np.nansum(candidates)

            # 以下的部分都是基于candidates_cnt的向量进行.
            # risk_matrix = risk_factor[candidates]
            x = exog[candidates]
            w0 = mp.weight[candidates]

            if any(holding_suspend):
                for ix, _ in enumerate(holding_suspend):
                    if _:
                        if any(np.isnan(exog[ix])):
                            self.logger.warn("Holding %s have nan factors %s" % (
                            mp.data_manager.codes[ix], column[np.isnan(exog[ix]).ravel()]))
                        if any(np.isnan(risk_factor[ix])):
                            self.logger.warn("Holding %s have nan factors %s" % (
                            mp.data_manager.codes[ix], risk_column[np.isnan(risk_factor[ix]).ravel()]))


            # constraint： weights < 1 - holding_suspend_sum
            constraints = [cp.sum(w) == 1 - holding_suspend_sum]

            # constraint: suspend locked
            weight_locked = (candidates & ~non_suspend)[candidates]
            if np.sum(weight_locked) >= 1:
                constraints += [w[weight_locked] == w0[weight_locked]]

            # constraint:for the non suspend, single_max constraint
            constraints += [w[~weight_locked] <= single_max]

            # constraint:for the non suspend, weight > 0
            constraints += [w[~weight_locked] >= 0]

            # 3. 行业风格暴露约束,相对benchmark上界约束
            risk_condition = kwargs.get("risk_condition", {"up": {}, "down": {}})

            # constraint: risk expo control , ceil
            for k, v in risk_condition['up'].items():
                col_index = risk_column == k
                expo = risk_factor[candidates][:, col_index]
                ceil = benchmark_expo[col_index] + v
                constraints += [cp.sum(cp.multiply(np.ravel(expo), -w)) >= -ceil]

            # constraint:risk expo control, floor
            for k, v in risk_condition['down'].items():
                col_index = risk_column == k
                expo = risk_factor[candidates][:, col_index]
                floor = benchmark_expo[col_index] - v
                constraints += [cp.sum(cp.multiply(np.ravel(expo), w)) >= floor]

            try:
                # transaction cost terms
                as_of_date = mp.date
                z = w - w0

                all_spread = mp.data_manager.get_bar(date=as_of_date, columns=["trade_spread_0935_1000"],
                                                     codes=mp.data_manager.codes)["trade_spread_0935_1000"]
                all_trade_price = mp.data_manager.get_bar(date=as_of_date, columns=["trade_price_0935_1000_n"],
                                                          codes=mp.data_manager.codes)["trade_price_0935_1000_n"]
                all_amount = mp.data_manager.get_bar(date=as_of_date, columns=["amount"], codes=mp.data_manager.codes)[
                    "amount"]
                all_tcost_sigma = \
                mp.data_manager.get_bar(date=as_of_date, columns=["pct_std22"], codes=mp.data_manager.codes)[
                    "pct_std22"]
                all_a = tc_a * all_spread / all_trade_price

                #transaction cost: first term coefficient
                a = all_a[candidates]

                tcost_sigma = all_tcost_sigma[candidates]
                # transaction cost: second term coefficient
                c1 = tcost_sigma / np.sqrt(all_amount[candidates] / total_value)

                # missing transaction cost,use default : 0.003
                ix = np.isnan(a) | np.isnan(c1) | np.isinf(c1)
                if ix.sum() > 0:
                    self.logger.info("%s missing transaction cost" % ix.sum())
                a[ix] = 0.003
                c1[ix] = 0.0

                # transaction cost: first term
                exp1 = cp.multiply(a, cp.abs(z))

                # transaction cost: second term
                power = tc_power
                exp2 = tc_b * cp.multiply(c1, cp.abs(z) ** power)

                # transaction cost: third term
                exp3 = tc_c * z

                tcost_expr = exp1 + exp2 + exp3
                tcost_expr = cp.sum(tcost_expr)

                # predicted return term
                pred_returnp_expr = cp.sum(cp.multiply(stock_return[candidates], w))

                assert (pred_returnp_expr.is_concave())

                # risk term
                """
                self.expression = cvx.sum_squares(cvx.multiply(
                np.sqrt(locator(self.idiosync, t).values), wplus)) + \
                cvx.quad_form((wplus.T * locator(self.exposures, t).values.T).T,
                              locator(self.factor_Sigma, t).values)
                """
                risk_expr = 2 * lambdax * cp.sum(cp.quad_form((w.T * x).T, cov))

                assert (risk_expr.is_convex())

                for el in constraints:
                    assert (el.is_dcp())

                prob = cp.Problem(cp.Maximize(pred_returnp_expr - risk_expr - tcost_expr), constraints)

                prob.solve(solver=cp.ECOS)

                if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                    user_log.info("status:{}".format(prob.status))
                    # user_log.info("w : {}".format(w.value))
                    user_log.info("sum(w):{}".format(np.sum(w.value)))

                    target_weight = np.full(stock_return.size, 0, dtype=np.double)
                    target_weight[holding_suspend] = mp.weight[holding_suspend]
                    target_weight[candidates] = np.round(w.value, 6)

                    # check expo
                    # user_log.info("max(w):{}", np.max(w.value))
                    # user_log.info("min(w):{}", np.min(w.value))
                    #
                    # import pandas as pd
                    # expo = pd.DataFrame()
                    #
                    # diff = risk_factor[candidates].T.dot(w.value) - benchmark_expo
                    # expo["factor"] = risk_column
                    # expo["diff"] = diff
                    # expo["abs"] = expo["diff"].abs()
                    # expo = expo.sort_values(by="abs", ascending=False)
                    # user_log.info(expo.head(50))

                    return target_weight,

                else:
                    user_log.info("status: {}".format(prob.status))
                    user_log.warning("optim failed at top n={} ,continue n+300".format(n))
                    n += 300

            except Exception as e:
                import traceback
                traceback.print_exc()
                break

        target_weight = mp.weight
        return target_weight,
