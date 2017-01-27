#!/usr/bin/python
import asyncio
import glob
import os
import asyncpg

queries = dict()

async def load_queries(path):
    """Prepares a folder of SQL files as SQL statements.

    :param path: Path to the `.sql` files.
    :type path: str
    """
    # load from queries/
    queryfiles = glob.glob(path + "/*.sql")
    for fp in queryfiles:
        with open(fp, "r", encoding="utf-8-sig") as qfile:
            name = os.path.splitext(os.path.basename(fp))[0]
            queries[name] = qfile.read()
