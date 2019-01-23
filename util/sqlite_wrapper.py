# import sqlite3 as sqlite
from sqlalchemy import create_engine
import pandas as pd
#
# class SqliteWrapper(object):
#     def __init__(self, dbfile):
#         self.__conn = None
#         self.__cursor = None
#         self.dbfile = dbfile
#         self.connect()
#
#     def __del__(self):
#         self.close()
#
#     def connect(self):
#         self.close()
#         self.__conn = sqlite.connect(self.dbfile)
#         self.__cursor = self.__conn.cursor()
#
#     def close(self):
#         if self.__cursor:
#             self.__cursor.close()
#             self.__cursor = None
#
#         if self.__conn:
#             self.__conn.close()
#             self.__conn = None
#
#     def select(self, sql):
#         self.__cursor.execute(sql)
#         return self.__cursor.fetchall()
#
#     def execute(self, sql):
#         self.__cursor.execute(sql)
#         self.__conn.commit()
#
#     def execute_batch(self, sqls):
#         for sql in sqls:
#             self.__cursor.execute(sql)
#         self.__conn.commit()


DB_FILE = "s.db"

class SqliteWrapper2(object):
    def __init__(self, dbfile=DB_FILE):
        self.engine = create_engine('sqlite:///%s' % dbfile)

    def select(self, trade_date, strategy, item):
        """
        按Key在数据库取Value
        :param trade_date:
        :param strategy:
        :param item:
        :return:
        """
        itemstr = "','".join(item)
        qry = "select * from strategy_deriv where trade_date = '%s' and strategy = '%s' and item in ('%s')" % (trade_date, strategy, itemstr)
        return self.select_query(qry)

    def select_query(self, query):
        """
        执行SQL-SELECT-QUERY
        :param query:
        :return:
        """
        return pd.read_sql_query(query, self.engine)

    def insert_long_table(self, name, df: pd.DataFrame, **kwargs):
        """
        插入长表
        :param name:
        :param df:
        :param kwargs:
        :return:
        """

        # FIXME: 这里my_tmp应该是一个random的字符串
        df.to_sql('my_tmp', self.engine, if_exists='replace', index=False)

        conn = self.engine.connect()
        trans = conn.begin()

        try:
            # delete those rows that we are going to "upsert"
            self.engine.execute(
                "delete from '%s' WHERE trade_date || strategy || item in (select trade_date || strategy || item from my_tmp)" % name)
            trans.commit()

            # insert changed rows
            df.to_sql(name, self.engine, if_exists="append", index=False)
        except Exception as e:
            trans.rollback()
            print(e)
            raise

    def insert_wide_table(self, name, df: pd.DataFrame, **kwargs):
        """
        插入宽表
        :param name:
        :param df:
        :param kwargs:
        :return:
        """
        newdf = pd.melt(df, id_vars=["trade_date", "strategy"], var_name="item")
        self.insert_long_table(name, newdf, **kwargs)

#
# if __name__ == "__main__":
#     s = SqliteWrapper2("s.db")
#     # d = pd.DataFrame({"trade_date": ["20190104", "20190104"],
#     #                   "strategy": ["test", "test2"],
#     #                   "item": "return",
#     #                   "value": [0.15, -0.05]})
#     # s.insert_long_table("strategy_deriv", d, if_exists="append")
#     #
#     # d = pd.DataFrame({"trade_date": ["20190104", "20190103"],
#     #                   "strategy": ["test", "test2"],
#     #                   "return": [0.11, -0.05]})
#     #
#     # s.insert_wide_table("strategy_deriv", d, if_exists="append")
#
#     a = s.select_query("select * from strategy_deriv where strategy='swing6f'")
#     print(a)
#
