# -*- coding: utf-8 -*-
# @author: scrat

import numpy as np
from cvxopt import solvers, matrix
from collider.data.sensor import Sensor


class OptimizationStockWeight(Sensor):

    @property
    def output_variables(self):
        return ["targetWeight"]

    def do(self, date, mp, **kwargs):

        # region 读入参数, 对应输入的数据

        # 优化器的参数
        lambdax = kwargs.get("lambdax", 1)  # lambda = 0.5?
        tc = kwargs.get("tc", 0.003)  # 手续费假定
        n = kwargs.get("top", 200)  # 前n个股票进入优化器
        single_max = kwargs.get("single_max", 0.02)  # 个股最大权重

        # benchmark weight
        weight_index = kwargs.get("benchmark_weight","weight_index_500")

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
            stock_return[np.any(np.isnan(exog), axis=1)] = np.nan

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

            # # candidates没有nan
            # candidates = (mp.weight > 0) | top_flag
            # # 最近一期的因子或者风格缺失的可能性如下: top_flag中因为可以有stock_return, 所以风格因子和alpha因子
            # # 都不会缺失; 持仓的股票是有可能有缺失的，比如停牌很久.
            # candidates &= ~np.any(np.isnan(exog), axis=1)
            # candidates &= ~np.any(np.isnan(risk_factor), axis=1)
            # candidates &= ~np.isnan(sp)

            candidates = top_flag.copy()  # 在top_flag中的肯定是有predicted_stock_return的，所以数据肯定不缺失
            # 在candidates中去掉其他nan的情况
            # case 1. special_risk为nan
            # candidates &= ~np.isnan(sp)

            # 在candidates中还需要加入以下的情况
            # case 1. 持仓 且 有stock_return (即数据不缺失）
            candidates |= (mp.weight > 0) & (~np.isnan(stock_return))

            # 持仓 且 停牌 且 数据缺失
            holding_suspend = (mp.weight > 0) & (is_suspend == 1) & (np.isnan(stock_return))
            holding_suspend_sum = np.sum(mp.weight[holding_suspend])

            candidates_cnt = np.nansum(candidates)

            # endregion

            # 以下的部分都是基于candidates_cnt的向量进行.
            # risk_matrix = risk_factor[candidates]
            x = exog[candidates]
            w0 = mp.weight[candidates]

            if any(holding_suspend):
                for ix, _ in enumerate(holding_suspend):
                    if _:
                        if any(np.isnan(exog[ix])):
                            self.logger.warn("Holding %s have nan factors %s" % (mp.data_manager.codes[ix], column[np.isnan(exog[ix]).ravel()]))
                        if any(np.isnan(risk_factor[ix])):
                            self.logger.warn("Holding %s have nan factors %s" % (mp.data_manager.codes[ix], risk_column[np.isnan(risk_factor[ix]).ravel()]))

            r = stock_return[candidates]
            sp_diag = np.diagflat(sp[candidates])

            # region 构造等式约束
            A_list = []
            b_list = []

            # 等式约束:
            # 1. sum权重的 = 1
            A_list.append(np.ones(shape=(1, candidates_cnt), dtype=np.double))
            b_list.append(np.array([1 - holding_suspend_sum]))
            # b_list.append(np.array([1]))

            # 2. 停牌股票的权重锁定(有持仓不能动，没持仓不能买)
            weight_locked = (candidates & ~non_suspend)[candidates]
            if any(weight_locked):
                weight_locked_cnt = np.nansum(weight_locked)  # 需要锁定的股票个数
                a_mat = np.zeros(shape=(weight_locked_cnt, candidates_cnt), dtype=np.double)
                for i, j in zip(range(weight_locked_cnt), np.arange(candidates_cnt)[weight_locked]):
                    a_mat[i, j] = 1
                A_list.append(a_mat)
                b_list.append(w0[weight_locked])  # 停牌股票权重

            # endregion

            # region 构造不等式条件
            G_list = []
            h_list = []

            # 1. 对于不停牌的股票，个股权重不能大于single_max
            wmax = np.full(candidates_cnt, single_max)

            # 个股支持最大权重不超过wmax
            neq_left_1 = np.eye(candidates_cnt)
            neq_right_1 = wmax
            G_list.append(neq_left_1[~weight_locked])
            h_list.append(neq_right_1[~weight_locked])

            # 2. 在long-only的前提下，个股权重非负
            neq_left_2 = -np.eye(candidates_cnt)
            neq_right_2 = np.zeros(candidates_cnt)
            G_list.append(neq_left_2[~weight_locked])
            h_list.append(neq_right_2[~weight_locked])

            # 3. 行业风格暴露约束,相对benchmark上界约束
            risk_condition = kwargs.get("risk_condition", {"up": {}, "down": {}})

            # 构造
            for k, v in risk_condition['up'].items():
                col_index = risk_column == k
                w = risk_factor[candidates][:, col_index]
                G_list.append(w.T)
                h_list.append(np.array([benchmark_expo[col_index] + v]))

            # 4. 行业风格暴露约束，相对指数下届约束
            # 风格约束
            for k, v in risk_condition['down'].items():
                col_index = risk_column == k
                w = risk_factor[candidates][:, col_index]
                G_list.append(-w.T)
                h_list.append(np.array([- benchmark_expo[col_index] + v]))

            # endregion

            # region 考虑交易费用的两步优化
            # z = np.maximum(wmax - w0, w0)
            #
            # commission_part1_left = np.eye(candidates_cnt)
            # commission_part1_right = z + w0
            #
            # commission_part2_left = -np.eye(candidates_cnt)
            # commission_part2_right = z - w0
            #
            # G_list.append(commission_part1_left)
            # h_list.append(commission_part1_right)
            #
            # G_list.append(commission_part2_left)
            # h_list.append(commission_part2_right)

            q = matrix(-stock_return[candidates])
            P = matrix(2 * lambdax * (np.dot(np.dot(x, cov), x.T) + sp_diag))
            A = matrix([matrix(a) for a in A_list], tc="d")
            b = matrix([matrix(b) for b in b_list], tc="d")
            G = matrix([matrix(g) for g in G_list], tc="d")
            h = matrix([matrix(h) for h in h_list], tc="d")

            try:
                res = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b, **kwargs)
                # kktsolver="ldl",
                # options={"show_progress": False, "maxiters": maxiters})
                if res["status"] != "optimal":
                    self.logger.warn("Stage 1 optimization failed at top %s, %s" % (n, date))

                    if n == non_nan_cnt:
                        raise Exception("non_nan_cnt(%s) is few at %s." % (n, date))
                    n += 300
                else:
                    self.logger.trace("Stage 1 optimization succeed at top %s" % n)

                    weights = np.ravel(res["x"])

                    # 第二次优化约束条件,
                    # 等式约束条件不变,
                    # 收益算上成本
                    cost = np.where((weights - w0) > 0, tc, -tc)
                    q = matrix(-r + cost, tc="d")


                    # 对于第一次求解判断为买入的票限制其权重为【w1,wmax】
                    # 对于第一次求解判断为卖出的票限制其权重为【0，w1】
                    '''
                        neq_left_1 = np.eye(candidates_cnt)
                        neq_right_1 = wmax
                        G_list.append(neq_left_1[~weight_locked])
                        h_list.append(neq_right_1[~weight_locked])
                    '''
                    # 第一次求解为卖出情况
                    opt_neq_left_1 = np.eye(candidates_cnt)
                    opt_neq_right_1 = w0

                    sell_codes = cost < 0

                    G_list.append(opt_neq_left_1[sell_codes])
                    h_list.append(opt_neq_right_1[sell_codes])

                    # 第一次求解为买入情况
                    opt_neq_left_2 = -np.eye(candidates_cnt)
                    opt_neq_right_2 = -w0

                    G_list.append(opt_neq_left_2[~sell_codes])
                    h_list.append(opt_neq_right_2[~sell_codes])



                    G = matrix([matrix(g) for g in G_list], tc="d")
                    h = matrix([matrix(h) for h in h_list], tc="d")
                    res = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b, **kwargs)

                    if res["status"] == "optimal":
                        self.logger.trace("Stage 2 optimization succeed at top %s" % n)
                        success = True
                    else:
                        if n == non_nan_cnt:
                            raise Exception("non_nan_cnt(%s) is few at %s." % (n, date))

                        self.logger.warn("Stage 2 optimization failed at top %s, %s" % (n, date))
                        n += 300

            except Exception as e:
                self.logger.exception(e)
                break

            # endregion

        if success:
            # fill to the fixed length
            target_weight = np.full(stock_return.size, 0, dtype=np.double)
            target_weight[holding_suspend] = mp.weight[holding_suspend]
            target_weight[candidates] = np.round(res["x"], 6).ravel()
        else:
            # 优化失败，持仓不变
            self.logger.warn("No optimize solution, keep holding. %s" % date)
            target_weight = mp.weight

        return target_weight,
