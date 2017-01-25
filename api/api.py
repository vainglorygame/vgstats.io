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

import uwsgi  # only available inside uwsgi
from uwsgidecorators import *


route = aiohttp_route_decorator.RouteCollector()


@timer(60, target="spooler")  # every minute
# FIXME: not working because of worker override
async def recrawl():
    print("getting recent matches")

    db = database.Database()
    await db.connect("postgres://vgstats@localhost/vgstats")
    api = crawler.Crawler()
    try:
        last_match_update = await db.meta("last_match_update")
    except KeyError:
        last_match_update = datetime.datetime(1, 1, 1).isoformat()
        await db.meta("last_match_update", last_match_update)
    await db.meta("last_match_update", datetime.datetime.now().isoformat())
    matches = await api.matches_since(last_match_update)
    await db.upsert("na", matches, True)


async def matches(request):
    print(await db.select("SELECT data->'attributes'->>'name' as name FROM player WHERE CAST(data->'attributes'->'stats'->>'winStreak' AS integer) > 3"))
    return

@route("/status")
async def status(request):
    resp = json.dumps({"version": "0.1.0"})
    uwsgi.signal(17)
    return aiohttp.web.Response(text=resp)

async def init_uwsgi(loop, fd):
    app = aiohttp.web.Application(loop=loop)
    route.add_to_router(app.router)
    srv = await loop.create_server(app.make_handler(),
                                   sock=socket.fromfd(fd,
                                                      socket.AF_INET,
                                                      socket.SOCK_STREAM))
    print("asyncio server started on uWSGI {0}".format(uwsgi.version))
    return srv


loop = asyncio.get_event_loop()
for fd in uwsgi.sockets:
    loop.run_until_complete(init_uwsgi(loop, fd))
uwsgi.accepting()
loop.run_forever()
