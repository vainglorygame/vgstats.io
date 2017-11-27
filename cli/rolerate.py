#!/usr/bin/python

import os
import json
import asyncio
import asyncpg
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


class RoleClassifier(object):
    """A DNN that classifies a participant in one of the three roles."""
    def __init__(self):
        self.model = None
        self._rolemap = ["captain", "lane", "jungle"]
        self._numfeatures = len(
            ["flare", "for", "crucible", "trap", "treads",
             "contraption", "protector", "dragonblood",
             "junglecs", "lances"])
        self.modeldir = "/tmp/vgstats-tf-role-model"
        self._pool = None

    async def connect(self, dbstring):
        """Connect to the database."""
        self._pool = await asyncpg.create_pool(dbstring)  # concurrency support

        # create layout
        # maps participant -> role jsonb
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS enhanced_participant" +
                    "(participant_id TEXT PRIMARY KEY, roles JSONB)")

    async def _get_sample(self, size="ALL", offset=0, identified=False):
        """Return a data set from the database.
        :param size: (optional) The number of items to get. Defaults to `All`.
        :type size: int or str
        :param offset: (optional) The SQL OFFSET parameter.
        :type offset: int or str
        :param identified: (optional) Whether to also return ids.
                           Defaults to `False`.
        :type identified: bool
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return await conn.fetch(
                """
                SELECT
                -- captain
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1038_Item_Flare*', '0')::int AS flare,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1045_Item_FountainOfRenewal*', '0')::int AS for,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1046_Item_Crucible*', '0')::int AS crucible,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1054_Item_ScoutTrap*', '0')::int AS trap,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1056_Item_WarTreads*', '0')::int AS treads,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1057_Item_AtlasPauldron*', '0')::int AS atlas,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1079_Item_Contraption*', '0')::int AS contraption,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1084_Item_ProtectorContract*', '0')::int AS protector,
                COALESCE(data->'attributes'->'stats'->'itemUses'->>'*1085_Item_DragonbloodContract*', '0')::int AS dragonblood,
                -- jungle
                (data->'attributes'->'stats'->>'jungleKills')::int AS junglecs,
                -- lane
                (data->'attributes'->'stats'->>'nonJungleMinionKills')::int AS lanecs
                """
             + ("""
                -- meta
                , id
                """ if identified else "") +
                """
                FROM participant
                LIMIT """ + str(size) + " OFFSET " + str(offset))

    async def _get_stats(self):
        """Return the average number of fountains, jungle cs and lane cs
           across all roles."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return (await conn.fetch(
                """
                SELECT
                  COUNT(*),
                  SUM((data->'attributes'->'stats'->'itemUses'->>'*1045_Item_FountainOfRenewal*')::int)::float
                    /
                  COUNT(*) AS for,
                  SUM((data->'attributes'->'stats'->>'jungleKills')::int)::float
                    /
                  COUNT(*) AS junglecs,
                  SUM((data->'attributes'->'stats'->>'nonJungleMinionKills')::int)::float
                    /
                  COUNT(*) AS lanecs
                FROM participant
                """))[0]

    async def _estimate(self, sample):
        """Take an input sample and classifies into the three roles
        based on a formula."""
        stats = await self._get_stats()

        estimate = []
        for samp in sample:
            scores = {"captain": 0, "lane": 0, "jungle": 0}

            # Assume that vital for lane and jungle is respective CS
            # and for roam fountain usage.
            # We take the average of each stat across all roles,
            # calculate the normalized distance of the player to the average,
            # and assume that the furthest distance is what the player played.
            if samp["lanecs"] > 0:
                scores["lane"] += samp["lanecs"] / stats["lanecs"]
            if samp["junglecs"] > 0:
                scores["jungle"] += samp["junglecs"] / stats["junglecs"]
            if samp["for"] > 0:
                scores["captain"] += samp["for"] / stats["for"]

            probably = max(scores, key=scores.get)
            estimate.append(probably)

        return estimate

    async def _model_setup(self):
        """Sets up a model that takes the `num_features` features as input."""
        feature_columns = [
            tf.contrib.layers.real_valued_column(
                "", dimension=self._numfeatures)
        ]

        self.model = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[5],  # http://stats.stackexchange.com/a/1097
            n_classes=3,  # "captain" "jungle" "lane"
            model_dir=self.modeldir,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=2
            )
        )

    async def train(self):
        """Train a DNN on a static, algorithmic guess."""
        await self._model_setup()

        if os.path.isdir(self.modeldir):
            print("already trained, not training again")
            return

        # get & split db data
        sample = await self._get_sample()
        testsize = len(sample) // 10  # 10%
        guess = await self._estimate(sample)
        guess = [self._rolemap.index(k) for k in guess]

        sample = np.array(sample)
        guess = np.array(guess)

        train_sample, test_sample = sample[:testsize], sample[testsize:]
        train_est, test_est = guess[:testsize], guess[testsize:]

        print(train_sample, train_est)
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            test_sample, test_est,
            every_n_steps=10
        )  # for debugging

        self.model.fit(
            x=train_sample, y=train_est,
            steps=40,
            batch_size=500,
            monitors=[validation_monitor]
        )
        # try with 500, at 30 steps we're at about 90-95% accuracy
        # it doesn't change significantly at ~100 (98.1%)
        #
        # We never want 100% accuracy!
        # That a captain uses his fountain more than the average player
        # is just a guess. The DNN should generalize, so it correlates
        # other features we feed it (vision, contracts) and weights them.
        # Once we have that, the network can rate a player
        # and give scores based on that.

    async def classify(self, sample, only_best=False):
        """Classify a data set.

        :param only_best: (optional) Return the predicted result
                          instead of a dict of propabilities.
        :type only_best: bool
        :return: Prediction results.
        :rtype: list of dict or list
        """
        samplearr = np.array(sample)
        if only_best:
            guesses = list(
                self.model.predict(
                    samplearr,
                    batch_size=500,
                    as_iterable=True
                ))
            return [self._rolemap[g] for g in guesses]
        else:
            guesses = list(
                self.model.predict_proba(
                    samplearr,
                    batch_size=500,
                    as_iterable=True
                ))
            # map role as key to probability as value
            return [dict(zip(self._rolemap, g)) for g in guesses]

    async def classify_db(self):
        """Classify all data in the data base and insert."""
        sample = await self._get_sample(identified=True)

        # split sample (with participant ids) into data and ids
        data = np.array(
            [[val for key, val in s.items() if key != "id"] for s in sample]
        )
        ids = [s["id"] for s in sample]
        # let the DNN do the work
        predicts = await self.classify(data)

        # convert numpy->dict->json for db
        predicts = [
            json.dumps({
                k: float(v) for k, v in p.items()
            }) for p in predicts
        ]
        dbdata = list(zip(ids, predicts))
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """INSERT INTO enhanced_participant(participant_id, roles)
                       VALUES ($1, $2)
                       ON CONFLICT (participant_id) DO
                       UPDATE SET data=$2
                    """, dbdata)


async def main():
    classifier = RoleClassifier()
    await classifier.connect("postgres://vgstats@localhost/vgstats")
    # TODO train only for the latest patch.
    await classifier.train()
    await classifier.classify_db()

asyncio.get_event_loop().run_until_complete(main())
