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


def plot_curves(inpaths, outpath, legend, colors, cols=4, budget=1e6):
  print('Loading runs:')
  runs = []
  for filename in inpaths:
    loaded = json.loads(pathlib.Path(filename).read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < budget - 1e4:
        print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  borders = np.arange(0, budget, 1e4)

  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  fig, ax = plt.subplots(figsize=(4.5, 2))
  for j, (method, label) in enumerate(legend.items()):
    relevant = [run for run in runs if run['method'] == method]
    if not relevant:
      print(f'No runs found for method {method}.')
    # Average within each time bucket.
    binned_xs, binned_ys = [], []
    for run in relevant:
      xs, ys = binning(run['xs'], run['reward'], borders, np.nanmean)
      binned_xs.append(xs)
      binned_ys.append(ys)
    xs = np.concatenate(binned_xs)
    ys = np.concatenate(binned_ys)
    # Compute mean and stddev over seeds.
    means = binning(xs, ys, borders, np.nanmean)[1]
    stds = binning(xs, ys, borders, np.nanstd)[1]
    # Plot line and shaded area.
    kwargs = dict(alpha=0.2, linewidths=0, color=colors[j], zorder=10 - j)
    ax.fill_between(borders[1:], means - stds, means + stds, **kwargs)
    ax.plot(borders[1:], means, label=label, color=colors[j], zorder=100 - j)

  ax.axhline(y=22, c='#888888', ls='--', lw=1)
  ax.text(6.2e5, 18, 'Optimal', c='#888888')

  ax.set_title('Crafter with Rewards')
  ax.set_xlim(0, budget)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  ax.xaxis.get_offset_text().set_visible(False)
  ax.set_xlabel('Million Env Steps')
  ax.set_ylabel('Episode Reward')
  ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))

  fig.tight_layout(rect=(0, 0, 0.55, 1))
  fig.legend(bbox_to_anchor=(0.52, 0.54), loc='center left', frameon=False)

  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


inpaths = [
    'runs/crafter-reward-dreamerv2.json',
    'runs/crafter-reward-rainbow.json',
    'runs/crafter-reward-ppo.json',
    'runs/crafter-reward-efficient_rainbow.json',
    'runs/crafter-reward-rnd.json',
    'runs/crafter-noreward-random.json',
]
legend = {
    'dreamerv2': 'DreamerV2',
    'rainbow': 'Rainbow',
    'ppo': 'PPO',
    'efficient_rainbow': 'Efficient Rainbow',
    'rnd': 'RND',
    'random': 'Random',
}
colors = ['#377eb8', '#4daf4a', '#984ea3', '#c68628', '#b54c38', '#785c56']
plot_curves(inpaths, 'results/curves.pdf', legend, colors)
