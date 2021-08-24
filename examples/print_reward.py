import json
import pathlib
import warnings

import numpy as np


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs, ys = np.array(xs), np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty borders become NaN.
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


def print_reward(indir, budget=5e6):
  print('Loading runs:')
  runs = []
  for filename in pathlib.Path(indir).glob('**/*.json'):
    loaded = json.loads(filename.read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < 0.99 * budget:
        print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  print('')
  methods = sorted(set(run['method'] for run in runs))
  borders = np.arange(0, budget, 1e5)

  print('Final episode reward:')
  for method in methods:
    means = []
    for run in runs:
      if run['method'] == method:
        binned = binning(run['xs'], run['score'], borders, np.nanmean)[1]
        means.append(binned[-1])
    mean = np.mean(means)
    std = np.std(means)
    print(f'{method:<15} {mean:6.2f} ± {std:.2f}')


print_reward('runs')
