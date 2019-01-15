#coding:utf8
import pandas as pd
import numpy as np
from collider.utils.logger import user_log
import copy
import os

from collider.utils.decompose.mod import AttributeAnaMod
from common.configure import read_configure

bundle_path = read_configure(name="test")['bundle_path']
config = {
     "data_bundle_path":bundle_path,
     "benchmark":"000905.SH"
}

ds = AttributeAnaMod(config).DS
pool_name = "pool_01_final_f1"
forward_return_name = "forward_return_5_f1"

def load_fakedvalue(a):
    return  a * 0.1 + np.random.normal(scale=0.02, size=4000)

def cal_hotcatch(factor_topN = 500,factors = [] ,ret_topN = 1000,dates=None):
    """
     计算top组，catch top catch ratio，返回  dataframe
    :param factor_topN:
    :param factors:
    :param ret_topN:
    :param dates:
    :return:
    """
    result = []
    for date in dates:

        pool = ~np.isnan(ds.get_bar(date=date,columns=[pool_name])[pool_name])
        ret = np.array(ds.get_bar(date=date ,columns=[forward_return_name])[forward_return_name])
        row = []
        for fac in factors:
            v =  np.array(ds.get_bar(date=date,columns=[fac])[fac])
            if fac == "forward_return_5_f1":
                v = load_fakedvalue(v)
            ret[~pool] = np.nan
            v[~pool] = np.nan

            code_a = ds.codes[np.argsort(-ret)[:ret_topN]]
            code_b = ds.codes[np.argsort(-v)[:factor_topN]]

            intersec = set(code_a).intersection(set(code_b))
            ratio = len(intersec)*1.0/len(code_b)
            row.append(ratio)

        result.append(row)

    cols = []
    for i in factors:
        if i == "forward_return_5_f1":
            cols.append("faked_" + i )
        else:
            cols.append(i)

    df = pd.DataFrame(result,columns=cols)
    df["trade_date"] = dates

    return df[["trade_date"] + cols]


def load_top_ret(factors=[],topN=500,weighted_type=1,dates=[]):

    """
       计算top组收益，
       weighted_type = 1 ,等权
       weighted_type = k/sum(1:N) 加权
    :param factors:
    :param topN:
    :param weighted_type:
    :param dates:
    :return:
    """
    if weighted_type == 1:
        weg = np.full(topN,1.0/topN)
    if weighted_type == 2:
        a = np.arange(1,topN+1,1)
        weg =  a/np.sum(a)

    result = []

    for date in dates:
        data = ds.get_bar(date=date,columns=[pool_name,forward_return_name])
        pool = ~np.isnan(data[pool_name])
        ret = np.array(data[forward_return_name])
        ret[~pool] = np.nan

        row = []
        for fac in factors:
             fac_data = np.array(ds.get_bar(date=date,columns=[fac])[fac])
             fac_data[~pool] = np.nan
             code_index = np.argsort(-fac_data)[:topN]
             v = np.nansum(ret[code_index] * weg[::-1])
             row.append(v)

        result.append(row)

    cols = []
    for i in factors:
        if i == "forward_return_5_f1":
            cols.append("faked_" + i)
        else:
            cols.append(i)

    result = np.cumsum(np.array(result),axis=0)
    df = pd.DataFrame(result ,columns=cols)
    df["trade_date"] = dates

    return df[["trade_date"] + cols]

if __name__ == "__main__":

    startdate = "20170101"
    end_date = "20181230"
    dates = [i for i in sorted(list(ds.dates)) if i >= startdate and i <= end_date]

    factor_topN = 500
    factors = ["predicted_stock_return_f1","flow_estimation_fitted_f1","forward_return_5_f1"]
    ret_topN = 500

    #df = cal_hotcatch(factor_topN=500,factors=factors,ret_topN=ret_topN,dates=dates)

    #df.to_csv("data/hotcatch_500.csv",index=False)

    df = load_top_ret(factors=factors,dates=dates)

    df.to_csv("data/group1_return.csv",index=False)












