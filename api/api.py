#!/usr/bin/python

import datetime
import asyncio
import socket
import sys
import json
import aiohttp.web
import aiohttp_route_decorator

import database
import crawler
import queries


route = aiohttp_route_decorator.RouteCollector(prefix="/api")
db = database.Database()


# TODO use logging module instead of print
async def crawl_region(region):
    """Gets some matches from a region and inserts them
       until the DB is up to date."""
    api = crawler.Crawler()

    while True:
        try:
            last_match_update = (await db.select(
                """
                SELECT data->'attributes'->>'createdAt' AS created
                FROM match
                WHERE data->'attributes'->>'shardId'='""" + region + """'
                ORDER BY data->'attributes'->>'createdAt' DESC LIMIT 1
                """)
            )[0]["created"]
        except:
            last_match_update = "2017-02-05T01:01:01Z"

        print(region + " fetching matches after " + last_match_update)

        # wait for http requests
        matches = await api.matches_since(last_match_update,
                                          region=region,
                                          params={"page[limit]": 50})
        if len(matches) > 0:
            print(region + " got new data items: " + str(len(matches)))
        else:
            print(region + " got no new matches.")
            return
        # insert asynchronously in the background
        await db.upsert(matches, True)


async def recrawl():
    """Gets the latest matches from all regions every 5 minutes."""
    print("getting recent matches")

    # TODO: insert API version (force update if changed)
    # TODO: create database indices
    # get or put when the last crawl was executed

    # crawl and upsert
    for region in ["na", "eu"]:
        # fire workers
        asyncio.ensure_future(crawl_region(region))

    await asyncio.sleep(300)
    asyncio.ensure_future(recrawl())


@route("/matches")
async def api_matches(request):
    data = (await db.select(queries.queries["recent-matches"]))[0]["data"]
    return aiohttp.web.Response(text=str(data))

@route("/winrates")
async def api_winrates(request):
    data = (await db.select(queries.queries["hero-winrates"]))[0]["data"]
    return aiohttp.web.Response(text=str(data))

@route("/status")
async def api_status(_):
    resp = json.dumps({"version": "0.1.0"})
    return aiohttp.web.Response(text=resp)


loop = asyncio.get_event_loop()
loop.run_until_complete(db.connect("postgres://vgstats@localhost/vgstats"))
loop.run_until_complete(queries.load_queries("api/queries/"))
loop.create_task(recrawl())
app = aiohttp.web.Application(loop=loop)
route.add_to_router(app.router)
app.router.add_static("/web", "web-dev")  # development web frontend
aiohttp.web.run_app(app)
