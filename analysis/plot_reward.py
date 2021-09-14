import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common


def plot_reward(inpaths, outpath, legend, colors, cols=4, budget=1e6):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  borders = np.arange(0, budget, 1e4)

  fig, ax = plt.subplots(figsize=(4.5, 2.3))
  for j, (method, label) in enumerate(legend.items()):
    relevant = [run for run in runs if run['method'] == method]
    if not relevant:
      print(f'No runs found for method {method}.')
    # Average within each time bucket.
    binned_xs, binned_ys = [], []
    for run in relevant:
      xs, ys = common.binning(run['xs'], run['reward'], borders, np.nanmean)
      binned_xs.append(xs)
      binned_ys.append(ys)
    xs = np.concatenate(binned_xs)
    ys = np.concatenate(binned_ys)
    # Compute mean and stddev over seeds.
    means = common.binning(xs, ys, borders, np.nanmean)[1]
    stds = common.binning(xs, ys, borders, np.nanstd)[1]
    # Plot line and shaded area.
    kwargs = dict(alpha=0.2, linewidths=0, color=colors[j], zorder=10 - j)
    ax.fill_between(borders[1:], means - stds, means + stds, **kwargs)
    ax.plot(borders[1:], means, label=label, color=colors[j], zorder=100 - j)

  ax.axhline(y=22, c='#888888', ls='--', lw=1)
  ax.text(6.2e5, 18, 'Optimal', c='#888888')

  ax.set_title('Crafter Reward')
  ax.set_xlim(0, budget)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  ax.grid(alpha=0.3)
  ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))

  fig.tight_layout(rect=(0, 0, 0.55, 1))
  fig.legend(bbox_to_anchor=(0.52, 0.54), loc='center left', frameon=False)

  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


inpaths = [
    'scores/crafter_reward-dreamerv2.json',
    'scores/crafter_reward-rainbow.json',
    'scores/crafter_reward-ppo.json',
    'scores/crafter_noreward-random.json',
]
legend = {
    'dreamerv2': 'DreamerV2',
    'rainbow': 'Rainbow',
    'ppo': 'PPO',
    'random': 'Random',
}
colors = ['#377eb8', '#4daf4a', '#984ea3', '#6a554d']
plot_reward(inpaths, 'plots/reward.pdf', legend, colors)
