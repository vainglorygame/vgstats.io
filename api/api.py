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


async def recrawl():
    """Gets the latest matches and inserts them into the database."""
    print("getting recent matches")
    api = crawler.Crawler()

    # TODO: insert API version (force update if changed)
    # TODO: create database indices
    # get or put when the last crawl was executed

    # crawl and upsert
    for region in ["na", "eu"]:
        try:
            last_match_update = (await db.select(
                """
                SELECT data->'attributes'->>'createdAt' AS created 
                FROM match
                WHERE data->'attributes'->>'shardId'='""" + region + """'
                ORDER BY data->'attributes'->>'createdAt' DESC LIMIT 1
                """)
            )[0]["created"]
        except IndexError:
            last_match_update = "2017-01-01T01:01:01Z"

        matches = await api.matches_since(last_match_update, region=region)
        if len(matches) > 0:
            print(region + " got a lot new data items: " + str(len(matches)))
        else:
            print(region + " got no new matches.")
        await db.upsert(matches, True)

    asyncio.ensure_future(recrawl_soon())

async def recrawl_soon():
    """Calls `recrawl` after 60 seconds."""
    print("crawler sleeping, Zzzzzz…")
    await asyncio.sleep(60)
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
loop.run_until_complete(queries.load_queries("queries/"))
loop.create_task(recrawl())
app = aiohttp.web.Application(loop=loop)
route.add_to_router(app.router)
app.router.add_static("/web", "../web-dev")  # development web frontend
aiohttp.web.run_app(app)
