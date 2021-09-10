import json
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


def plot_grid(
    inpath, outpath, color, budget=int(1e6), cols=4, size=(2, 1.8)):

  print('Loading runs:')
  runs = []
  loaded = json.loads(pathlib.Path(inpath).read_text())
  for run in [loaded] if isinstance(loaded, dict) else loaded:
    print(f'- {run["method"]} seed {run["seed"]}', flush=True)
    if run['xs'][-1] < budget - 1e4:
      print(f'  Contains only {run["xs"][-1]} steps!')
    runs.append(run)
  borders = np.arange(0, budget, 1e4)
  keys = ('reward', 'length')
  keys += tuple(k for k in runs[0].keys() if k.startswith('achievement_'))

  rows = len(keys) // cols
  fig, axes = plt.subplots(
      nrows=rows, ncols=cols, figsize=(size[0] * cols, size[1] * rows))

  for ax, key in zip(axes.flatten(), keys):
    ax.set_title(key.replace('achievement_', '').replace('_', ' ').title())
    xs = np.concatenate([run['xs'] for run in runs])
    ys = np.concatenate([run[key] for run in runs])

    binxs, binys = binning(xs, ys, borders, np.nanmean)
    ax.plot(binxs, binys, color=color)

    mins = binning(xs, ys, borders, np.nanmin)[1]
    maxs = binning(xs, ys, borders, np.nanmax)[1]
    ax.fill_between(binxs, mins, maxs, linewidths=0, alpha=0.2, color=color)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))

    if maxs.max() == 0:
      ax.set_ylim(-0.05, 1.05)

  fig.tight_layout()
  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


plot_grid(
    'runs/crafter_reward-dreamerv2.json',
    'results/grid-dreamerv2.pdf', '#377eb8')

plot_grid(
    'runs/crafter_reward-ppo.json',
    'results/grid-ppo.pdf', '#5fc35d')

plot_grid(
    'runs/crafter_reward-rainbow.json',
    'results/grid-rainbow.pdf', '#984ea3')

plot_grid(
    'runs/crafter_noreward-unsup_rnd.json',
    'results/grid-unsup_rnd.pdf', '#bf3217')

plot_grid(
    'runs/crafter_noreward-unsup_plan2explore.json',
    'results/grid-unsup_plan2explore.pdf', '#de9f42')

plot_grid(
    'runs/crafter_noreward-random.json',
    'results/grid-random.pdf', '#6a554d')
