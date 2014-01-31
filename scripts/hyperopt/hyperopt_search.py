#!/usr/bin/env python

from __future__ import print_function

import sys
import math

import hyperopt
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
             hp.uniform('histogramBias', 0.0, 0.6),
             )
    return space


def get_exp(mongodb_url, database, exp_key):
    trials = MongoTrials('mongo://%s/%s/jobs' % (mongodb_url, database), exp_key=exp_key)
    space = get_space()
    return trials, space

def show(mongodb_url, db, exp_key):
      print ("Get trials, space...")
      trials, space = get_exp(mongodb_url, db, exp_key)
      print ("Get bandit...")
      bandit = hyperopt.Bandit(expr=space, do_checks=False)
      print ("Plotting...")
      # from IPython.core.debugger import Tracer; Tracer()()
      best_trial = trials.best_trial
      values = best_trial['misc']['vals']
      loss = best_trial['result']['loss']
      true_loss = best_trial['result']['true_loss']
      print ("values: ", values)
      print ("loss: ", loss)
      print ("true_loss: ", true_loss)
      hyperopt.plotting.main_plot_history(trials)
      hyperopt.plotting.main_plot_vars(trials, bandit, colorize_best=3)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <mongodb-url> <database> <experiment> [show]" % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    mongodb_url, database, exp_key = sys.argv[1:4]

    if len(sys.argv) == 5:
        show(mongodb_url, database, exp_key)
        sys.exit(0)

    trials, space = get_exp(mongodb_url, database, exp_key)
    best = fmin(fn=math.sin, space=space, trials=trials, algo=tpe.suggest, max_evals=1000)
    print("best: %s" % (repr(best)))
