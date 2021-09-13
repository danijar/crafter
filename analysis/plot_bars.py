import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt


def plot_bars(inpaths, outpath, legend, colors, budget=1e6, sort=False):

  print('Loading runs:')
  runs = []
  for filename in inpaths:
    loaded = json.loads(pathlib.Path(filename).read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < budget - 1e4:
        print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
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

  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  if sort:
    first = list(legend.keys())[0]
    order = np.argsort(-np.nanmean(percents[methods.index(first)], 0), -1)
    percents = percents[:, :, order]
    tasks = np.array(tasks)[order].tolist()

  fig, ax = plt.subplots(figsize=(7, 3))
  centers = np.arange(len(tasks))
  width = 0.7
  for index, (method, label) in enumerate(legend.items()):
    heights = np.nanmean(percents[methods.index(method)], 0)
    pos = centers + width * (0.5 / len(methods) + index / len(methods) - 0.5)
    color = colors[index]
    ax.bar(pos, heights, width / len(methods), label=label, color=color)

  names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
      axis='x', which='both', width=14, length=0.8, direction='inout')
  ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
  ax.set_xticks(centers + 0.0)
  ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')

  ax.set_ylabel('Success Rate (%)')
  ax.set_yscale('log')
  ax.set_ylim(0.01, 100)
  ax.set_yticks([0.01, 0.1, 1, 10, 100])
  ax.set_yticklabels('0.01 0.1 1 10 100'.split())

  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.legend(
      loc='upper center', ncol=10, frameon=False, borderpad=0, borderaxespad=0)

  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


inpaths = [
    'scores/crafter_reward-dreamerv2.json',
    'scores/crafter_reward-ppo.json',
    'scores/crafter_reward-rainbow.json',
]
legend = {
    'dreamerv2': 'DreamerV2',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',
}
colors = ['#377eb8', '#5fc35d', '#984ea3']
plot_bars(inpaths, 'results/bars-reward.pdf', legend, colors)

inpaths = [
    'scores/crafter_noreward-unsup_rnd.json',
    'scores/crafter_noreward-unsup_plan2explore.json',
    'scores/crafter_noreward-random.json',
]
legend = {
    'unsup_rnd': 'RND',
    'unsup_plan2explore': 'Plan2Explore',
    'random': 'Random',
}
colors = ['#bf3217', '#de9f42', '#6a554d']
plot_bars(inpaths, 'results/bars-noreward.pdf', legend, colors)
