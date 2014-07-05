#!/usr/bin/env python
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################

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
