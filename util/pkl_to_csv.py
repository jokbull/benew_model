# %%
import numpy as np
import os
import pandas as pd
# from common.data_source_from_bundle import data_source
# DataSource = data_source()
# codes = np.append(DataSource.codes, np.full(4000-len(DataSource.codes), "XXXXXX"))


import pickle


def read_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f).data


def to_csv(df, file):
    with open(file, "a") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def read_coefficient(root, scenario, trade_date, **kwargs):
    data_file_name = os.path.join(root, "cache", "%s.OLS.coefficient" % scenario, "%s.pkl" % trade_date)
    data = read_pkl(data_file_name)

    factor_name_file_name = os.path.join(root, "cache", "%s.OLS.factor_name" % scenario, "%s.pkl" % trade_date)
    factor_name = read_pkl(factor_name_file_name)

    return pd.DataFrame({"factor_name": factor_name, "trade_date": trade_date, "coefficient": data})


def read_data(root, scenario, dates, func, **kwargs):
    if isinstance(dates, list) or isinstance(dates, np.ndarray):
        result = pd.concat([read_data(root, scenario, date, func, **kwargs) for date in dates])
        return result
    else:
        result = func(root, scenario, dates, **kwargs)
        return result


if __name__ == "__main__":
    from common.data_source_from_bundle import trading_dates
    td = trading_dates()

    dates = td.get_trading_dates("20100310", "20190111")

    root_path = "/Users/scrat/workspace/benew_model"
    scenarios = "flow_estimation"
    result = read_data(root_path, scenarios, dates, read_coefficient)
    to_csv(result, os.path.join(root_path, "coef.csv"))
#
# #%%
# # Method 2: 从OLS.model读入数据
# import pandas as pd
# trade_date_list = os.listdir(os.path.join(ROOT, "cache", "flow_estimation.OLS.date"))
# trade_date_list.sort()
# for trade_date_file in trade_date_list:
#     trade_date = trade_date_file[:8]
#     for scen in scenarios:
#         model = read_pkl(os.path.join(ROOT, "cache", "%s.OLS.model" % scen, trade_date_file))
#         factor_name = read_pkl(os.path.join(ROOT, "cache", "%s.OLS.factor_name" % scen, trade_date_file))
#         pool = read_pkl(os.path.join(ROOT, "cache", "%s.OLS.pool" % scen, trade_date_file))
#         exposure = model.exog
#         for i, fac_name in enumerate(factor_name):
#             new_fac_name = "%s_%s" % (fac_name, scen)
#             data = pd.DataFrame({
#                 'wind_code': codes[pool],
#                 'trade_date': trade_date,
#                 new_fac_name: exposure[:, i]
#             })
#             data.dropna(inplace=True)
#             to_csv(data[['trade_date', 'wind_code', new_fac_name]], file=os.path.join(ROOT, "out", "%s_%s.csv" % (fac_name, scen)))
#
