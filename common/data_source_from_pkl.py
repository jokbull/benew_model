import os
import pickle


def read_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f).data


def load_data_from_pkl(trade_date, factor_name, return_type="numpy", **kwargs):
    """

    :param trade_date: 只能取一天的
    :param factor_name:
    :param return_type:
    :param kwargs:
    :return:
    """
    root = kwargs.pop("root")
    scenario = kwargs.pop("scenario", "")

    if factor_name == "predicted_stock_return":
        func = read_predicted_stock_return
    else:
        raise Exception("")
    result = func(root=root, scenario=scenario, trade_date=trade_date, **kwargs)

    if return_type == "numpy":
        return result
    else:
        raise NotImplementedError("Not support return_type %s" % return_type)


def read_predicted_stock_return(root, scenario, trade_date, **kwargs):
    data_file_name = os.path.join(root, "cache", "%s.predictionStockReturn.stockReturn" % scenario,
                                  "%s.pkl" % trade_date)
    data = read_pkl(data_file_name)
    return data


#
# def read_coefficient(root, scenario, trade_date, **kwargs):
#     data_file_name = os.path.join(root, "cache", "%s.OLS.coefficient" % scenario, "%s.pkl" % trade_date)
#     data = read_pkl(data_file_name)
#
#     factor_name_file_name = os.path.join(root, "cache", "%s.OLS.factor_name" % scenario, "%s.pkl" % trade_date)
#     factor_name = read_pkl(factor_name_file_name)
#
#     return pd.DataFrame({"factor_name": factor_name, "trade_date": trade_date, "coefficient": data})


if __name__ == "__main__":
    a = load_data_from_pkl("20190104", "predicted_stock_return",
                           root="/Users/scrat/workspace/benew_model/",
                           scenario="flow_prediction_ortho")
    print(a)
