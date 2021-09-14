import json
import pathlib
import warnings

import numpy as np


def load_runs(filenames, budget=1e6, verbose=True):
  verbose and print('')
  runs = []
  for filename in filenames:
    loaded = json.loads(pathlib.Path(filename).read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      message = f'Loading {run["method"]} seed {run["seed"]}'
      verbose and print(message, flush=True)
      if run['xs'][-1] < budget - 1e4:
        verbose and print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  verbose and print('')
  return runs


def compute_success_rates(runs, budget=1e6, sortby=None):
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))
  tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))
  percents = np.empty((len(methods), len(seeds), len(tasks)))
  percents[:] = np.nan
  for run in runs:
    episodes = (np.array(run['xs']) <= budget).sum()
    i = methods.index(run['method'])
    j = seeds.index(run['seed'])
    for key, values in run.items():
      if key in tasks:
        k = tasks.index(key)
        percent = 100 * (np.array(values[:episodes]) >= 1).mean()
        percents[i, j, k] = percent
  if isinstance(sortby, (str, int)):
    if isinstance(sortby, str):
      sortby = methods.index(sortby)
    order = np.argsort(-np.nanmean(percents[sortby], 0), -1)
    percents = percents[:, :, order]
    tasks = np.array(tasks)[order].tolist()
  return percents, methods, seeds, tasks


def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs, ys = np.array(xs), np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty buckets become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for start, stop in zip(borders[:-1], borders[1:]):
      left = (xs <= start).sum()
      right = (xs <= stop).sum()
      if left < right:
        value = reducer(ys[left:right])
      elif binned:
        value = {'nan': np.nan, 'last': binned[-1]}[fill]
      else:
        value = np.nan
      binned.append(value)
  return borders[1:], np.array(binned)
