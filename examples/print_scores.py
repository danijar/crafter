import collections
import json
import pathlib

import numpy as np


def print_scores(indir, legend, budget=5e6):

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

  percents = collections.defaultdict(dict)
  for task in tasks:
    for method in legend.keys():
      values = []
      for run in runs:
        if run['method'] == method:
          episodes = (np.array(run['xs']) <= budget).sum()
          achievements = np.array(run[task])[:episodes]
          percent = 100 * (achievements >= 1).mean()
          values.append(percent)
      percents[method][task] = np.mean(values)

  first = next(iter(legend.keys()))
  tasks = sorted(tasks, key=lambda task: -percents[first][task])
  legend = dict(reversed(legend.items()))
  cols = ''.join(f' & {k:<9}' for k in legend.values())
  print(r'\newcommand{\o}{\hphantom{0}}')
  print(f'{"Achievement":<20}' + cols + r' \\')
  print('')
  for task in tasks:
    if task.startswith('achievement_'):
      name = task[len('achievement_'):].replace('_', ' ').title()
    else:
      name = task.replace('_', ' ').title()
    print(f'{name:<20}', end='')
    for method in legend.keys():
      fmt = f'{percents[method][task]:.1f}'
      fmt = (r'\o' if len(fmt) < 4 else ' ') + fmt
      print(rf' & ${fmt}\%$', end='')
    print(r' \\')
  print('')

  print(f'{"Average":<20}', end='')
  for method in legend.keys():
    fmt = f'{np.mean(list(percents[method].values())):.1f}'
    fmt = (r'\o' if len(fmt) < 4 else ' ') + fmt
    print(rf' & ${fmt}\%$', end='')
  print(r' \\')


legend = {'dreamerv2': 'DreamerV2', 'ppo': 'PPO', 'random': 'Random'}
print_scores('runs', legend)
