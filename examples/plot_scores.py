import json
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt


def plot_bars(
    inpaths, outpath, legend, colors, budget=1e6, sort=False, ylim=None):

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
  # Geometric mean.
  with warnings.catch_warnings():  # Empty borders become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1

  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  legend = dict(reversed(legend.items()))

  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  mean = np.nanmean(scores, -1)
  std = np.nanstd(scores, -1)

  fig, ax = plt.subplots(figsize=(4, 3))
  centers = np.arange(len(legend))
  width = 0.7
  colors = list(reversed(colors[:len(legend)]))
  error_kw = dict(capsize=5, c='#000')
  ax.bar(centers, mean, yerr=std, color=colors, error_kw=error_kw)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
      axis='x', which='both', width=50, length=0.8, direction='inout')
  ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
  ax.set_xticks(centers + 0.0)
  ax.set_xticklabels(
      list(legend.values()), rotation=45, ha='right', rotation_mode='anchor')

  ax.set_ylabel('Crafter Score (%)')
  if ylim:
    ax.set_ylim(0, ylim)

  fig.tight_layout()
  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


inpaths = [
    'runs/crafter_reward-human.json',
    'runs/crafter_reward-dreamerv2.json',
    'runs/crafter_reward-ppo.json',
    'runs/crafter_reward-rainbow.json',
    'runs/crafter_noreward-unsup_rnd.json',
    'runs/crafter_noreward-unsup_plan2explore.json',
    'runs/crafter_noreward-random.json',
]
legend = {
    'human': 'Human Experts',
    'dreamerv2': 'DreamerV2',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',
    'unsup_rnd': 'RND\n(Unsup)',
    'unsup_plan2explore': 'Plan2Explore\n(Unsup)',
    'random': 'Random',
}
colors = [
    '#cccccc',
    '#377eb8', '#5fc35d', '#984ea3',
    '#bf3217', '#de9f42', '#6a554d',
]
plot_bars(inpaths, 'results/scores-human.pdf', legend, colors, ylim=100)

inpaths = [
    'runs/crafter_reward-dreamerv2.json',
    'runs/crafter_reward-ppo.json',
    'runs/crafter_reward-rainbow.json',
    'runs/crafter_noreward-unsup_rnd.json',
    'runs/crafter_noreward-unsup_plan2explore.json',
    'runs/crafter_noreward-random.json',
]
legend = {
    'dreamerv2': 'DreamerV2',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',
    'unsup_rnd': 'RND\n(Unsup)',
    'unsup_plan2explore': 'Plan2Explore\n(Unsup)',
    'random': 'Random',
}
colors = [
    '#377eb8', '#5fc35d', '#984ea3',
    '#bf3217', '#de9f42', '#6a554d',
]
plot_bars(inpaths, 'results/scores-agents.pdf', legend, colors, ylim=12)
