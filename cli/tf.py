#!/usr/bin/python

import os
import asyncio
import asyncpg
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class WinClassifier(object):
    """A Linear Regression that predicts win or loss based on KDA."""
    def __init__(self):
        self.model = None
        self.modeldir = "/tmp/vgstats-tf-win-model"
        self._pool = None
        self._numfeatures = len(["cs", "k", "d", "a"])

    async def connect(self, dbstring):
        """Connect to the database."""
        self._pool = await asyncpg.create_pool(dbstring)  # concurrency support

        # create layout
        # maps participant -> role jsonb
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS predicted_wins " +
                    "(participant_id TEXT PRIMARY KEY, data JSONB)")

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
                    (data->'attributes'->'stats'->>'winner')::bool AS win,
                    (data->'attributes'->'stats'->>'farm')::float AS cs,
                    (data->'attributes'->'stats'->>'kills')::int AS k,
                    (data->'attributes'->'stats'->>'deaths')::int AS d,
                    (data->'attributes'->'stats'->>'assists')::int AS a
                """
             + (", id" if identified else "") +
                "FROM participant " +
                "LIMIT " + str(size) + " OFFSET " + str(offset))

    def _model_setup(self):
        """Create the layers and the linear classifier."""
        feature_columns = [
            tf.contrib.layers.real_valued_column(
                "", dimension=self._numfeatures)
        ]

        self.model = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[],
            n_classes=2,  # "win" "loss"
            model_dir=self.modeldir,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=2
            )
        )


    async def train(self):
        self._model_setup()

        if os.path.isdir(self.modeldir):
            print("already trained, not training again")
            return

        sample = await self._get_sample(size=100000)  # out of 800 000

        # prepare
        data = np.array(
            [[val for key, val in s.items() if key != "win"] for s in sample]
        )
        results = np.array([int(s["win"]) for s in sample])  # loss: 0, win: 1

        # split into 90% train and 10% test
        testsize = len(data) // 10
        train_sample, test_sample = data[:testsize], data[testsize:]
        train_est, test_est = results[:testsize], results[testsize:]

        # do stuff.
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            test_sample, test_est,
            every_n_steps=10
        )  # for debugging

        self.model.fit(
            x=train_sample, y=train_est,
            steps=200, # TODO sample size, steps
            monitors=[validation_monitor]
        )

    async def classify(self, sample, only_best=False):
        inp = np.array(sample)
        if only_best:
            return list(self.model.predict(inp, as_iterable=True))
        else:
            guesses = list(self.model.predict_proba(
                inp, as_iterable=True))
            # map role as key to probability as value
            return [dict(zip(["loss", "win"], g)) for g in guesses]


async def main():
    classifier = WinClassifier()
    await classifier.connect("postgres://vgstats@localhost/vgstats")
    await classifier.train()
    sample = await classifier._get_sample(size=10)
    data = np.array(
        [[val for key, val in s.items() if key != "win"] for s in sample]
    )
    # TODO train batches/fixed number
    print(await(classifier.classify(data)))
#    await classifier.classify_db()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
