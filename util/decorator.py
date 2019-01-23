from util.sqlite_wrapper import SqliteWrapper2
from functools import lru_cache, wraps
import numpy as np
import pandas as pd


# https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays

def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


class cache(object):
    def __init__(self, dbfile=":memory:"):
        self.sqlite = SqliteWrapper2(dbfile)

    def __call__(self, func):  # 接受函数
        def wrapper(*args, **kwargs):
            # 检查数据库里是否存在 and kwargs.get("read", True) is True
            #     | --- Y： 返回数据库中的数据
            #     | --- N:  执行func进行计算
            #               如果kwargs.get("write", True) is True，那么就写入数据库
            flag = False
            if kwargs.get("read", True):
                try:
                    result_df = self.sqlite.select(kwargs["trade_date"], kwargs["strategy"], kwargs["item"])
                    if len(result_df.index) == len(kwargs["item"]):
                        flag = True
                        result = result_df.value.tolist()
                    else:
                        flag = False
                except KeyError:
                    flag = False
                except Exception:
                    raise

            if not flag:  # 数据库不存在
                result = func(*args, **kwargs)
            if kwargs.get("write", True) and flag is False:
                result_df = pd.DataFrame({
                    "trade_date": kwargs['trade_date'],
                    "strategy": kwargs['strategy'],
                    "item": kwargs['item'],
                    "value": result
                })
                self.sqlite.insert_long_table("strategy_deriv", result_df)
            return result

        return wrapper  # 返回函数
