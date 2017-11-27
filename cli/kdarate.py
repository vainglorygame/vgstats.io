#!/usr/bin/python

import os
import itertools
import json
import asyncio
import asyncpg
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


class ActorClassifier(object):
    """A DNN that classifies loss/win based on hero and items."""
    def __init__(self):
        self.model = None
        self.modeldir = "/tmp/vgstats-tf-actor-model"
        self._pool = None

        # DNN configuration
        self._categories = []
        self._continuous = ["kills", "deaths", "assists"]
        self._classes = ["loss", "win"]
        self._label = "win"

        # learn configuration
        self._steps = 200  # per batch
        self._max_batches = 20  # number of batches to train

        # database mappings
        self._paths = {
            "id": "id",
            "kills": "(data->'attributes'->'stats'->>'kills')::int",
            "deaths": "(data->'attributes'->'stats'->>'deaths')::int",
            "assists": "(data->'attributes'->'stats'->>'assists')::int",
#            "actor": "data->'attributes'->>'actor'",
            "win": "(data->'attributes'->'stats'->>'winner')::bool::int"
        }
#        for n in range(0, 6):
#            self._paths["item"+str(n)] = \
#            "COALESCE(data->'attributes'->'stats'->'items'->>" + str(n) +", '')"

        self._step = 0  # saves batch learning state

    async def _async_connect(self, dbstring):
        """Connect to the database."""
        # TODO with current synchronous implementation, pooling isn't needed
        self._pool = await asyncpg.create_pool(dbstring)

        # create layout
        # maps participant -> role jsonb
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    # TODO do this globally etc
                    "CREATE TABLE IF NOT EXISTS enhanced_participant_wip" +
                    "(participant_id TEXT PRIMARY KEY, actor JSONB)")

    def connect(self, dbstring):
        """See _async_connect."""
        asyncio.get_event_loop().run_until_complete(
            self._async_connect(dbstring))

    # TODO maybe don't use asyncpg
    async def _async_get_sample(self, size="ALL", offset=0):
        """Return a data set from the database.
        :param size: (optional) The number of items to get. Defaults to `All`.
        :type size: int or str
        :param offset: (optional) The SQL OFFSET parameter.
        :type offset: int or str
        :type identified: bool
        """
        query = "SELECT "
        elements = []
        for name, path in self._paths.items():
            elements.append(
                """
                ARRAY(SELECT
                  {3}
                  FROM participant
                  ORDER BY id
                  LIMIT {0} OFFSET {1}
                ) AS {2}
                """.format(size, offset, name, path))

        query += ", ".join(elements)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                return (await conn.fetch(query))[0]

    def _get_sample(self, size="ALL", offset=0):
        """See _async_get_sample."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_get_sample(size, offset))

    def _model_setup(self):
        """Sets up a model that takes the `num_features` features as input."""
        feature_columns = [
            tf.contrib.layers.sparse_column_with_hash_bucket(feat, 100)
            for feat in self._categories
        ]
        emb_columns = [
            tf.contrib.layers.embedding_column(
                sparse_id_column=col,
                dimension=5  # log_2(number of unique features)  TODO
            )
            for col in feature_columns
        ] + [
            tf.contrib.layers.real_valued_column(feat)
            for feat in self._continuous
        ]

        self.model = tf.contrib.learn.DNNClassifier(
            feature_columns=emb_columns,
            hidden_units=[2],  # http://stats.stackexchange.com/a/1097  TODO inp+outp / 2
            n_classes=len(self._classes),
            model_dir=self.modeldir,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=2
            )
        )

    def _toinput(self, sample, train=True):
        """Convert sample to Tensors.
        :param sample: Data dictionary.
        :type sample: dict
        :param train: (optional) Whether to return labels too.
        :type train: bool
        :return: features, label
        :rtype: tuple
        """
        continuous = {k: tf.constant(sample[k])
                      for k in self._continuous}
        categories = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(len(sample[k]))],
            values=sample[k],
            shape=[len(sample[k]), 1])
                    for k in self._categories}

        features = {**continuous, **categories}

        if train:
            label = tf.constant(sample[self._label])
            return features, label
        else:
            return features

    # TODO use asyncpg cursor https://magicstack.github.io/asyncpg/current/api/index.html#cursors
    def _more(self, batchsize=500, limit=None):
        """Fetches up to `limit` number of training items
           from the database in batches."""
        # TODO maybe you can use an iterator?
        if limit:
            if self._step > limit:
#               self._step = 0
                # TODO ????
                raise tf.errors.OutOfRangeError
        sample = self._get_sample(
            size=batchsize, offset=self._step)
        self._step += batchsize
        return self._toinput(sample, train=True)

    def train(self):
        """Train a DNN on a static, algorithmic guess."""
        self._model_setup()

        if os.path.isdir(self.modeldir):
            print("already trained, not training again")
            return

        # get one batch of testing data
        # TODO either terminate with `eval_steps` or with OutOfRangeError
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self._toinput(self._get_sample(size=1000, offset=101000), train=True),
            eval_steps=1,
            every_n_steps=20,
            early_stopping_metric="accuracy",
            early_stopping_metric_minimize=False,
            early_stopping_rounds=100
        )

        # validation monitor stops learning if the accuracy does not increase of 100 steps
        for _ in range(self._max_batches):
            self.model.fit(
                input_fn=lambda: self._more(),
                steps=self._steps,
                monitors=[validation_monitor]
            )

    def classify(self, sample, only_best=False):
        """Classify a data set.

        :param only_best: (optional) Return the predicted result
                          instead of a dict of propabilities.
        :type only_best: bool
        :return: Prediction results.
        :rtype: list of dict or list
        """
        if only_best:
            return self.model.predict(input_fn=lambda: self._toinput(sample, train=False))
        else:
            return self.model.predict_proba(input_fn=lambda: self._toinput(sample, train=False))

    async def classify_db(self):
        # TODO rewrite
        """Classify all data in the data base and insert."""
        sample = self._get_sample(identified=True)

        # split sample (with participant ids) into data and ids
        data = np.array(
            [[val for key, val in s.items() if key != "id"] for s in sample]
        )
        ids = [s["id"] for s in sample]
        # let the DNN do the work
        predicts = self.classify(data)

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
                    """INSERT INTO enhanced_participant_wip(participant_id, actor)
                       VALUES ($1, $2)
                       ON CONFLICT (participant_id) DO
                       UPDATE SET data=$2
                    """, dbdata)


def main():
    classifier = ActorClassifier()
    classifier.connect("postgres://vgstats@localhost/vgstats")
    # TODO train only for the latest patch.
    print("training")
    classifier.train()
    print("done training")
    sample = classifier._get_sample(size=10, offset=102000) # DEBUG
    print(sample)
    # TODO train batches/fixed number
    print("classifying debug data")
    d = classifier.classify(sample)
    for l in itertools.islice(d, 10):
        print(l)
    print(sample["win"])
    print("classified")
    #classifier.classify_db()

main()
