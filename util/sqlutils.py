# coding:utf8
# encoding:utf8
import pandas as pd
from collider.utils.logger import user_log
import pymysql
import time
import numpy as np


class SqlUtils():
    AVG = "avg"
    STD = "std"
    SUM = "sum"
    AVG_ABS = "avg(abs"
    GT_ABS_2 = 2

    def __init__(self, host="10.19.196.222", port=3306, user="Fz_prod_read", passwd="Fz_factor168_read",
                 db="benew_factor", charset='utf8'):
        """
         默认给了read权限mysql连接

        :param host:
        :param port:
        :param user:
        :param passwd:
        :param db:
        :param charset:
        """
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset
        self.conn = None
        self._conn()

        self.func = {
            "avg": "avg",
            "std": "std",
            "avg_abs": "avg(abs())"

        }

    """数据库连接和关闭"""

    def _conn(self):
        try:
            self.conn = pymysql.connect(host=self.host, user=self.user, password=self.passwd,
                                        database=self.db, port=self.port, charset=self.charset)
            return True
        except Exception as e:
            user_log.warning(e)
            return False

    def _re_conn(self, stime=3):
        """重试连接
        """
        status = True
        while status:
            try:
                # ping 校验连接是否异常
                self.conn.ping()
                status = False
            except:
                # 重新连接,成功退出
                print("Start reconnecting")
                if self._conn():
                    status = False
                time.sleep(stime)

    def close(self):
        self.conn.close()

    """
        ---数据操作---
    """

    def insert_feature(self, data, tablename, **kwargs):
        """

        data: dataframe 对象，取["trade_date","factor","item","value"]]4列
        :param data:
        :param tablename:
        :param kwargs:
        :return:
        """

        # 检查数据格式
        try:
            check_data = data[["trade_date", "factor", "item", "value"]]
        except Exception as e:
            user_log.error("data column incorrect,stop insert database")
            return False

        # 检查数据 inf 和 nan
        check_data = check_data.replace([np.inf, -np.inf], np.nan)
        check_data = check_data.where(check_data.notnull(), None)

        force = kwargs.get("force", False)

        user_log.info("insert into table {},data records {} ".format(tablename, len(check_data)))
        sql = "insert into " + tablename + "(trade_date,factor,item,value) values(%s,%s,%s,%s)"
        if force:
            sql = sql.replace("insert", "replace")

        fill_values = check_data.values.tolist()

        self._re_conn()

        try:
            cur = self.conn.cursor()

            cur.executemany(sql, fill_values)
            self.conn.commit()
        except Exception as e:
            user_log.warning(e)
            return False
        finally:
            self.close()
        return True

    def delete_feature(self, tablename, factors=[], features=[], **kwargs):

        """
         删除rows,
         如果 factors = [] ,则删除所有factors的记录
         如果 featrure = [] ,则删除所有feature的记录
         如果 同时为空，不进行删除

        :param tablename:
        :param factors:
        :param features:
        :param kwargs:
        :return:
        """

        if len(factors) == 0 and len(features) == 0:
            user_log.info("miss parameters")
            return False

        sql = "delete from " + tablename + " where "

        params = []

        if len(factors) > 0:
            sql += " factor in  %s and"
            params.append(factors)
        if len(features) > 0:
            sql += " item in %s and"
            params.append(features)
        sql = sql[:-4]

        self._re_conn()
        try:
            user_log.info("delete rows.....")
            cur = self.conn.cursor()
            cur.execute(sql, params)
            self.conn.commit()
            user_log.info("delete done")
        except Exception as e:
            user_log.warning(e)
            return False
        finally:
            self.close()

    def query(self, tablename, factors=[], features=[], **kwargs):

        """
        根据条件查询
        返回 dataframe:
        cols如下：
           [factor,trade_date,featere1 ,featrue2,featrue3]
        :param tablename:
        :param factors:
        :param features:
        :return:
        """
        startdate = kwargs.get("startdate", "20100101")
        enddate = kwargs.get("enddate", "22222222")

        sql = "select trade_date,factor,item,value from " + tablename + " where trade_date >= %s and trade_date <= %s "

        data = []

        for feature in features:
            sql1 = sql + " and item = %s"

            self._re_conn()

            try:
                print(sql1)
                cur = self.conn.cursor()
                cur.execute(sql1, (startdate, enddate, feature))

                results = cur.fetchall()

                self.conn.commit()

                results = pd.DataFrame(list(results), columns=['trade_date', 'factor', 'item', 'value'])
                results.index = [results["factor"], results["trade_date"]]
                if len(factors) > 0:
                    results = results.ix[factors]

                results[feature] = results["value"]
                data.append(results[feature])
            except Exception as e:
                user_log.warning(e)
                return False
            finally:
                self.close()

        user_log.info("query done,begin merge!")
        df = pd.concat(data, axis=1)
        df = df.reset_index(level=["factor", "trade_date"])

        return df

    def query_groupby(self, tablename, features={}, **kwargs):

        """
        查询，对features进行groupby 操作，
           例如：查询ic平均， 参数为features= {ic:SqlUtils.AVG}
        :param tabelename:
        :param features:
        :param kwargs:
        :return:
        """

        startdate = kwargs.get("startdate", "20120101")
        enddate = kwargs.get("enddate", "20222222")

        user_log.info("query start - end : {} - {} ".format(startdate, enddate))

        if len(features) == 0:
            user_log.warning("no features in paramter")
            return None

        result = []

        for feature in features.keys():
            for func in features[feature]:

                if type(func) == str:

                    count_close_parentheses = func.count("(") + 1
                    temp = [')'] * count_close_parentheses
                    sql = "select factor," + func + "(value" + "".join(temp) \
                          + " as " + feature + "_" + func.replace("(", "_") \
                          + " from " + tablename

                    sql += " where trade_date >= %s and trade_date  <= %s"
                    sql += " and item = %s"
                    sql += " group by factor"
                    user_log.info(sql)

                    self._re_conn()

                    columns = ["factor", feature + "_" + func.replace("(", "_")]

                    try:
                        cur = self.conn.cursor()
                        cur.execute(sql, (startdate, enddate, feature))
                        data = cur.fetchall()
                        self.conn.commit()

                        df = pd.DataFrame(list(data), columns=columns)

                        df.index = df["factor"]

                        result.append(df[columns[-1]])

                    except Exception as e:
                        user_log.warning(e)
                        user_log.warning("error in query_groupby {}".format(feature))
                    finally:

                        self.close()
                else:

                    columns = ["factor", feature + "_abs_gt_" + str(SqlUtils.GT_ABS_2)]

                    sql = "select factor, sum(if(abs(value) > %s,1,0))/count(value) "
                    sql += " as " + columns[-1]
                    sql += " from factor_main where "
                    sql += " trade_date >= %s and trade_date  <= %s  and item = %s group by factor"

                    user_log.info(sql)

                    self._re_conn()

                    try:
                        cur = self.conn.cursor()

                        cur.execute(sql, (SqlUtils.GT_ABS_2, startdate, enddate, feature))
                        data = cur.fetchall()
                        self.conn.commit()

                        df = pd.DataFrame(list(data), columns=columns)
                        df.index = df["factor"]
                        result.append(df[columns[-1]])

                    except Exception as e:
                        user_log.warning(e)
                        user_log.warning("error in query_groupby {}".format(feature))

                    finally:
                        self.close()

        df = pd.concat(result, axis=1)
        df = df.reset_index()

        return df


if __name__ == "__main__":
    utils = SqlUtils(
        host="localhost", port=3306,
        user="root",
        passwd="zengyilin",
        db="benew_factor",
        charset='utf8'
    )

    features = {
        "ic": [SqlUtils.AVG, SqlUtils.STD],
        "factor_return": [SqlUtils.AVG],
        "tvalue": [SqlUtils.AVG_ABS, SqlUtils.GT_ABS_2]
    }

    data = utils.query_groupby(tablename="factor_main", features=features)

    print(data.head())
