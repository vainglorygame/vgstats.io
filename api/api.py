#!/usr/bin/python

import asyncio
import database
import crawler
import datetime


async def main():
    db = database.Database()
    await db.connect("postgres://vgstats@localhost/vgstats")
    api = crawler.Crawler()
    try:
        await db.meta("last_match_update")
    except KeyError:
        await db.meta("last_match_update",
                      datetime.datetime(1, 1, 1).isoformat())
    matches = await api.matches_since(
        await db.meta("last_match_update"))
    await db.upsert("na", matches, True)
    print(await db.select("SELECT data->'attributes'->>'name' as name FROM player WHERE CAST(data->'attributes'->'stats'->>'winStreak' AS integer) > 3"))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
