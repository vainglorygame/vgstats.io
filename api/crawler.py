#!/usr/bin/python

import requests
import datetime

TOKEN = "aaa.bbb.ccc"
APIURL = "https://api.dc01.gamelockerapp.com/"

class Crawler(object):
    def __init__(self):
        """Sets constants."""
        self._apiurl = APIURL
        self._token = TOKEN
        self._lastquery = datetime.datetime(1, 1, 1)
        self._pagelimit = 50

    def _req(self, path, params=None):
        """Sends an API request and returns the response dict.

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
        http = requests.get(self._apiurl + path, headers=headers,
                            params=params)
        http.raise_for_status()
        return http.json()

    def matches(self, region="na", params=None):
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
        resp = []
        params["page[limit]"] = self._pagelimit
        params["page[offset]"] = 0
        while True:
            # go one page forward until 404
            try:
                json = self._req("shards/" + region + "/matches", params)
            except requests.exceptions.HTTPError:
                break
            resp += json["data"] + json["included"]
            params["page[offset]"] += params["page[limit]"]
        return resp

    def matches_new(self, region="na"):
        """Queries the API for new matches since the last query.

        :param region: see `matches`
        :type region: str
        """
        lastdate = self._lastquery.replace(microsecond=0).isoformat() + "Z"
        self._lastquery = datetime.datetime.now()
        return self.matches(region, {"filter[createdAt-start]": lastdate})
