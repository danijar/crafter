import json
import pathlib
import warnings

import numpy as np


def print_scores(inpaths, legend, budget=1e6, sort=False):

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
  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  means = np.nanmean(scores, -1)
  stds = np.nanstd(scores, -1)

  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  print('')
  print(r'\textbf{Method} & \textbf{Score} \\')
  print('')
  for method, mean, std in zip(legend.values(), means, stds):
    mean = f'{mean:.1f}'
    mean = (r'\o' if len(mean) < 4 else ' ') + mean
    print(rf'{method:<25} & ${mean} \pm {std:4.1f}\%$ \\')
  print('')


inpaths = [
    'scores/crafter_reward-dreamerv2.json',
    'scores/crafter_reward-ppo.json',
    'scores/crafter_reward-rainbow.json',
    'scores/crafter_noreward-unsup_rnd.json',
    'scores/crafter_noreward-unsup_plan2explore.json',
    'scores/crafter_noreward-random.json',
    'scores/crafter_reward-human.json',
]
legend = {
    'human': 'Human Experts',
    'dreamerv2': 'DreamerV2',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',
    'unsup_rnd': 'RND (Unsup)',
    'unsup_plan2explore': 'Plan2Explore (Unsup)',
    'random': 'Random',
}
print_scores(inpaths, legend)
