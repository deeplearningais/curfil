#!/usr/bin/env python

from __future__ import print_function

import sys
import math

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials


def get_space():
    space = (hp.quniform('numTrees', 1, 10, 1),
             hp.quniform('samplesPerImage', 10, 7500, 1),
             hp.quniform('featureCount', 10, 7500, 1),
             hp.quniform('minSampleCount', 1, 1000, 1),
             hp.quniform('maxDepth', 5, 25, 1),
             hp.quniform('boxRadius', 1, 127, 1),
             hp.quniform('regionSize', 1, 127, 1),
             hp.quniform('thresholds', 10, 60, 1),
             hp.uniform('histogramBias', 0.0, 0.0),
             )
    return space


def get_exp(mongodb_url, database, exp_key):
    trials = MongoTrials('mongo://%s/%s/jobs' % (mongodb_url, database), exp_key=exp_key)
    space = get_space()
    return trials, space


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <mongodb-url> <database> <experiment>" % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    mongodb_url, database, exp_key = sys.argv[1:]

    trials, space = get_exp(mongodb_url, database, exp_key)
    best = fmin(fn=math.sin, space=space, trials=trials, algo=tpe.suggest, max_evals=1000)
    print("best: %s" % (repr(best)))
