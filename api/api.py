#!/usr/bin/python

import database
import crawler

db = database.Database("dbname=vgstats user=vgstats")
api = crawler.Crawler()
#db.upsert("na", api.matches(params={"filter[playerNames]": "fallingcomets"}), True)
db.upsert("na", api.matches(), True)
print(db.select("SELECT data->'attributes'->>'name' as name FROM player WHERE CAST(data->'attributes'->'stats'->>'winStreak' AS integer) > 10"))
