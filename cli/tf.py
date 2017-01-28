#!/usr/bin/python
"""
Work in progress
 - Tensorflow Linear Regression to predict win/loss based on KDA
"""

import asyncio
import pandas
import tensorflow as tf

import api.database


db = api.database.Database()


async def get_winrate_sample():
    return await db.select(
        """
        SELECT 
            (data->'attributes'->'stats'->>'winner')::bool AS win,
            data->'attributes'->>'actor' AS hero,
            (data->'attributes'->'stats'->>'farm')::float AS cs,
            (data->'attributes'->'stats'->>'kills')::int AS k,
            (data->'attributes'->'stats'->>'deaths')::int AS d,
            (data->'attributes'->'stats'->>'assists')::int AS a
        FROM participant
        """
    )

def to_input_winrate(df):
    LABEL = "win"
    CATEGORIES = ["hero"]
    CONTINUOUS = ["cs", "k", "d", "a"]

    # convert constants to tensors
    conts = {k: tf.constant(df[k])
             for k in CONTINUOUS}
    # convert categories to sparse tensors
    cats = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(len(df[k]))],
        values=df[k],
        shape=[len(df[k]), 1])
            for k in CATEGORIES}

    df[LABEL] = [int(d) for d in df[LABEL]]  # bool to 0 / 1

    features = {**conts, **cats}
    label = tf.constant(df[LABEL])
    return features, label

async def train_winrate():
    data = await get_winrate_sample()  # list of dict
    trainlimit = int(len(data) * 0.8)

    # convert to dict of list
    train_sample = {}
    for it in data[:trainlimit]:
        for k, v in it.items():
            try:
                train_sample[k].append(v)
            except KeyError:
                train_sample[k] = [v]
    test_sample = {}
    for it in data[trainlimit:]:
        for k, v in it.items():
            try:
                test_sample[k].append(v)
            except KeyError:
                test_sample[k] = [v]

    # create input layers
    cs = tf.contrib.layers.real_valued_column("cs")
    k = tf.contrib.layers.real_valued_column("k")
    d = tf.contrib.layers.real_valued_column("d")
    a = tf.contrib.layers.real_valued_column("a")
    hero = tf.contrib.layers.sparse_column_with_hash_bucket("hero", hash_bucket_size=50)

    wide_columns = [
        hero, cs, k, d, a
    ]

    model = tf.contrib.learn.LinearClassifier(
        feature_columns=wide_columns
    )

    def train():
        return to_input_winrate(train_sample)
    def test():
        return to_input_winrate(test_sample)

    model.fit(input_fn=train, steps=200)
    results = model.evaluate(input_fn=test, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


tf.logging.set_verbosity(tf.logging.INFO)

loop = asyncio.get_event_loop()
loop.run_until_complete(db.connect("postgres://vgstats@localhost/vgstats"))
loop.run_until_complete(train_winrate())
