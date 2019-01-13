from collider.data.sensor import Sensor
from collider.data.message_package import MessagePackage

class GetPool(Sensor):


    @property
    def output_variables(self):
        return ["pool"]

    def do(self, date, mp: MessagePackage, **kwargs):

        pool_name = kwargs["pool_name"]

        as_of_date = mp.date
        next_date = mp.data_manager.trading_dates.get_next_trading_date(as_of_date)

        pool_codes_dict = mp.data_manager.get_bar(
            date=as_of_date,
            columns=[pool_name]
        )
        pool_index = pool_codes_dict[pool_name] == 1

        # 如果有基准，那么把基准的成分都加入pool
        if "benchmark_weight" in kwargs:
            benchmark = kwargs["benchmark_weight"]
            benchmark_weight = mp.data_manager.get_bar(date=as_of_date, columns=[benchmark])[benchmark]
            pool_index |= benchmark_weight > 0

        # 如果存在当前持仓
        if hasattr(mp, "weight"):
            pool_index |= mp.weight > 0

        # 去掉st股票, is_st比较特殊，可以认为可以取到第二天的
        st = mp.data_manager.get_bar(date=next_date, columns=["is_st"])["is_st"]
        is_st = st == 1
        pool_index &= (~is_st)

        # FIXME：blacklist
        # if self.blacklist:
        #     blacklist = self.data.setdefault("blacklist", None)
        #     # 黑名单要取交易日的,  池子是用前一日的信息
        #     black_dict = self.data_source.get_bar(
        #         date=self.data_source.trading_dates.get_next_trading_date(input), columns=blacklist)
        #
        #     for b in blacklist:
        #         np.bitwise_and(pool_index, black_dict[b] != 1, out=pool_index)

        return pool_index,