#!/usr/bin/python

import asyncio
import database
import crawler


async def main():
    db = database.Database()
    await db.connect("postgres://vgstats@localhost/vgstats")
    api = crawler.Crawler()
    matches = await api.matches()
    await db.upsert("na", matches, True)
    print(await db.select("SELECT data->'attributes'->>'name' as name FROM player WHERE CAST(data->'attributes'->'stats'->>'winStreak' AS integer) > 3"))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
