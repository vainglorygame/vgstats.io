#!/usr/bin/python

import asyncio
import json
import asyncpg


class Database(object):
    """Database wrapper class"""
    def __init__(self):
        self._pool = None

    async def connect(self, connstring):
        """Connects to the database.

        :param connstring: Connection string containing user and database.
        :type connstring: str
        """
        self._pool = await asyncpg.create_pool(connstring)

    async def upsert_type(self, obj, objtype, many=False):
        """Upserts an object of given `objtype` into the corresponding database.

        :param obj: Object to upsert.
        :type obj: dict or list
        :param objtype: Object type and table name.
        :type objtype: str
        :param many: (optional) Whether `obj` is a list
                     of objects of the same objtype.
        :type many: bool
        """
        if not many:
            obj = [obj]

        arr = [[
            (j["attributes"].get("shardId") or "") + j["id"],
            json.dumps(j)
        ] for j in obj]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("CREATE TABLE IF NOT EXISTS " + objtype +
                                   " (id TEXT PRIMARY KEY, data jsonb)")
                await conn.executemany("INSERT INTO " + objtype + " " +
                                       "VALUES ($1, $2) " +
                                       "ON CONFLICT (id) DO " +
                                       "UPDATE SET data=$2",
                                       arr)

    async def upsert(self, obj, many=False):
        """Upserts an object into the corresponding database.

        :param obj: Object to upsert.
        :type obj: dict
        :param many: (optional) Whether `obj` is a list
                     of objects.
        :type many: bool
        """
        if not many:
            obj = [obj]
        objectmap = dict()

        # figure out the type of each object and sort into map
        for j in obj:
            objtype = j["type"]
            if objtype not in objectmap:
                objectmap[objtype] = []
            objectmap[objtype].append(j)

        # execute bulk upsert for each type
        tasks = []
        for objtype, objects in objectmap.items():
            task = asyncio.ensure_future(
                self.upsert_type(objects, objtype, True))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def execute(self, query, args):
        """Runs an SQL statement.

        :param query: SQL query to execute.
        :type query: str
        :param args: Query arguments.
        :type args: list of objects
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(query, args)

    async def select(self, query, *args):
        """Returns the result of an SQL query.

        :param query: SQL query to execute.
        :type query: str
        :param args: Query arguments.
        :return: List of results.
        :rtype: list of dict
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return await conn.fetch(query, *args)
