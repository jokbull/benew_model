from util.sqlite_wrapper import SqliteWrapper
from functools import lru_cache, wraps
import numpy as np


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
        self.sqlite = SqliteWrapper(dbfile)

    def __call__(self, func):  # 接受函数
        def wrapper(*args, **kwargs):
            # FIXME
            # 检查数据库里是否存在 and kwargs.get("read", True) is True
            #     | --- Y： 返回数据库中的数据
            #     | --- N:  执行func进行计算
            #               如果kwargs.get("write", True) is True，那么就写入数据库
            if kwargs.get("read", True):
                pass  # try to read, return bool

            if True:  # 数据库存在
                result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            if kwargs.get("write", True):
                pass
            return result

        return wrapper  # 返回函数
