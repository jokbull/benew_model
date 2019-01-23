import sqlite3 as sqlite

class SqliteWrapper(object):
    def __init__(self, dbfile):
        self.__conn = None
        self.__cursor = None
        self.dbfile = dbfile
        self.connect()

    def __del__(self):
        self.close()

    def connect(self):
        self.close()
        self.__conn = sqlite.connect(self.dbfile)
        self.__cursor = self.__conn.cursor()

    def close(self):
        if self.__cursor:
            self.__cursor.close()
            self.__cursor = None

        if self.__conn:
            self.__conn.close()
            self.__conn = None

    def select(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()

    def execute(self, sql):
        self.__cursor.execute(sql)
        self.__conn.commit()

    def execute_batch(self, sqls):
        for sql in sqls:
            self.__cursor.execute(sql)
        self.__conn.commit()
