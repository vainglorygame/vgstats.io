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
        last_match_update = await db.meta("last_match_update")
    except KeyError:
        last_match_update = datetime.datetime(1, 1, 1).isoformat()
        await db.meta("last_match_update", last_match_update)
    print("getting new matches since " + last_match_update)
    await db.meta("last_match_update", datetime.datetime.now().isoformat())
    matches = await api.matches_since(last_match_update)
    print("inserting data into database")
    await db.upsert("na", matches, True)
    print("calculating stats")
    print(await db.select("SELECT data->'attributes'->>'name' as name FROM player WHERE CAST(data->'attributes'->'stats'->>'winStreak' AS integer) > 3"))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
