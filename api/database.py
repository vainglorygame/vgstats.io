#!/usr/bin/python

import psycopg2
import psycopg2.extras

class Database(object):
    """Database wrapper class"""
    def __init__(self, connstring):
        """Connects to the database.

        :param connstring: Connection string containing user and database.
        :type connstring: str
        """
        self._connection = psycopg2.connect(connstring)
        self._c = self._connection.cursor()

    def upsert_type(self, shard, json, objtype, many=False):
        """Upserts an object of given `objtype` into the corresponding database.

        :param shard: Region in which an object's id is unique.
        :type shard: str
        :param json: Object to upsert.
        :type json: dict
        :param objtype: Object type and table name.
        :type objtype: str
        :param many: (optional) Whether `json` is an array
                     of objects of the same objtype.
        :type many: bool
        """
        if not many:
            json = [json]

        arr = [{
            "id": shard + j["id"],
            "data": psycopg2.extras.Json(j)
        } for j in json]

        self._c.execute("CREATE TABLE IF NOT EXISTS " + objtype +
                        " (id TEXT PRIMARY KEY, data json)")

        self._c.executemany("INSERT INTO " + objtype + " " +
                            "VALUES (%(id)s, %(data)s) " +
                            "ON CONFLICT (id) DO " +
                            "UPDATE SET data=%(data)s",
                            arr)
        self._connection.commit()

    def upsert(self, shard, json, many=False):
        """Upserts an object into the corresponding database.

        :param shard: Region in which an object's id is unique.
        :type shard: str
        :param json: Object to upsert.
        :type json: dict
        :param many: (optional) Whether `json` is an array
                     of objects.
        :type many: bool
        """
        if not many:
            json = [json]
        objectmap = dict()

        # figure out the type of each object and sort into map
        for j in json:
            objtype = j["type"]
            if objtype not in objectmap:
                objectmap[objtype] = []
            objectmap[objtype].append(j)

        # execute bulk upsert for each type
        for objtype, objects in objectmap.items():
            self.upsert_type(shard, objects, objtype, True)

    def select(self, query):
        """Returns the result of an SQL query.

        :param query: SQL query to execute.
        :type query: str
        :return: List of results.
        :rtype: list of dict
        """
        self._c.execute(query)
        return self._c.fetchall()
