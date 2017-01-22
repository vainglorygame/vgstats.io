#!/usr/bin/python

import asyncio
import datetime
import aiohttp

TOKEN = "aaa.bbb.ccc"
APIURL = "https://api.dc01.gamelockerapp.com/"


class Crawler(object):
    def __init__(self):
        """Sets constants."""
        self._apiurl = APIURL
        self._token = TOKEN
        self._lastquery = datetime.datetime(1, 1, 1)
        self._pagelimit = 50

    async def _req(self, session, path, params=None):
        """Sends an API request and returns the response dict.

        :param session: aiohttp client session.
        :type session: :class:`aiohttp.ClientSession`
        :param path: URL path.
        :type path: str
        :param params: (optional) Request parameters.
        :type params: dict
        :return: API response.
        :rtype: dict
        """
        headers = {
            "Authorization": "Bearer " + self._token,
            "X-TITLE-ID": "semc-vainglory",
            "Accept": "application/vnd.api+json",
            "Content-Encoding": "gzip"
        }
        async with session.get(self._apiurl + path, headers=headers,
                               params=params) as response:
            if response.status != 200:
                return None
            return await response.json()

    async def matches(self, region="na", params=None):
        """Queries the API for matches and their related data.

        :param region: (optional) Region where the matches were played.
                       Defaults to "na" (North America).
        :type region: str
        :param params: (optional) Additional filters.
        :type params: dict
        :return: Processed API response
        :rtype: list of dict
        """
        if params is None:
            params = dict()
        params["page[limit]"] = self._pagelimit
        params["page[offset]"] = 0
        batchsize = 100  # FIXME We don't know how long we can paginate

        tasks = []
        data = []
        # fire a lot of http requests
        async with aiohttp.ClientSession() as session:
            for _ in range(0, batchsize):
                params["page[offset]"] += params["page[limit]"]
                task = asyncio.ensure_future(
                    self._req(
                        session,
                        "shards/" + region + "/matches",
                        params))
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            for res in results:
                if res is None:
                    continue
                data += res["data"] + res["included"]

        return data

    async def matches_new(self, region="na"):
        """Queries the API for new matches since the last query.

        :param region: see `matches`
        :type region: str
        """
        lastdate = self._lastquery.replace(microsecond=0).isoformat() + "Z"
        self._lastquery = datetime.datetime.now()
        return self.matches(region, {"filter[createdAt-start]": lastdate})
