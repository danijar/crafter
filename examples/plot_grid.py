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


def plot_grid(indir, legend, colors, cols=4, budget=5e6):
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
  tasks = sorted([x for x in runs[0] if x.startswith('achievement_')])
  tasks = ['score', 'length'] + tasks
  borders = np.arange(0, budget, 1e5)

  print('Plotting:')
  rows = len(tasks) // cols
  fig, axes = plt.subplots(
      nrows=rows, ncols=cols, figsize=(2 * cols, 1.8 * rows))
  for i, (task, ax) in enumerate(zip(tasks, axes.flatten())):
    ax.set_title(task.replace('achievement_', '').replace('_', ' ').title())
    maxes = []
    for j, (method, label) in enumerate(legend.items()):
      if i > 0: label = None
      relevant = [run for run in runs if run['method'] == method]
      print(f'- {task} {method} ({len(relevant)} seeds)', flush=True)
      xs = np.concatenate([run['xs'] for run in relevant])
      ys = np.concatenate([run[task] for run in relevant])
      means = binning(xs, ys, borders, np.nanmean)[1]
      mins = binning(xs, ys, borders, np.nanmin)[1]
      maxs = binning(xs, ys, borders, np.nanmax)[1]
      kwargs = dict(alpha=0.2, linewidths=0, color=colors[j], zorder=10 - j)
      ax.fill_between(borders[1:], mins, maxs, **kwargs)
      ax.plot(borders[1:], means, label=label, color=colors[j], zorder=100 - j)
      maxes.append(maxs)
    if np.nanmax(maxes) < 1:
      ax.set_ylim(-0.05, 1.05)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.08)
  fig.legend(loc='lower center', ncol=100, frameon=False)
  return fig


filename = pathlib.Path('results/grid-dreamerv2.pdf')
filename.parent.mkdir(exist_ok=True, parents=True)
legend = {'dreamerv2': 'DreamerV2'}
colors = ['#377eb8']
plot_grid('runs', legend, colors).savefig(filename)
print(f'Saved {filename}\n')

filename = pathlib.Path('results/grid-ppo.pdf')
filename.parent.mkdir(exist_ok=True, parents=True)
legend = {'ppo': 'PPO'}
colors = ['#4daf4a']
plot_grid('runs', legend, colors).savefig(filename)
print(f'Saved {filename}\n')

filename = pathlib.Path('results/grid-random.pdf')
filename.parent.mkdir(exist_ok=True, parents=True)
legend = {'random': 'Random'}
colors = ['#a65628']
plot_grid('runs', legend, colors).savefig(filename)
print(f'Saved {filename}\n')
