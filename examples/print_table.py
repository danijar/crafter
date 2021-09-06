import collections
import json
import pathlib

import numpy as np


def print_table(inpaths, legend, budget=1e6):

  print('Loading runs:')
  runs = []
  for filename in inpaths:
    loaded = json.loads(pathlib.Path(filename).read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < budget - 1e4:
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
  cols = ''.join(f' & {k:<13}' for k in legend.values())
  print(r'\newcommand{\o}{\hphantom{0}}')
  print(r'\newcommand{\b}[1]{\textbf{#1}}')
  print('')
  print(f'{"Achievement":<20}' + cols + r' \\')
  print('')
  for task in tasks:
    if task.startswith('achievement_'):
      name = task[len('achievement_'):].replace('_', ' ').title()
    else:
      name = task.replace('_', ' ').title()
    print(f'{name:<20}', end='')
    best = max(percents[m][task] for m in legend.keys())
    for method in legend.keys():
      value = percents[method][task]
      bold = value >= 0.95 * best and value > 0
      fmt = f'{value:.1f}'
      fmt = (r'\o' if len(fmt) < 4 else ' ') + fmt
      fmt = rf'\b{{{fmt}}}' if bold else f'   {fmt} '
      print(rf' & ${fmt}\%$', end='')
    print(r' \\')
  print('')

  print(f'{"Average":<20}', end='')
  averages = {m: np.mean(list(percents[m].values())) for m in legend.keys()}
  best = max(averages.values())
  for method, average in averages.items():
    fmt = f'{average:.1f}'
    fmt = (r'\o' if len(fmt) < 4 else ' ') + fmt
    fmt = rf'\b{{{fmt}}}' if bold else f'   {fmt} '
    print(rf' & ${fmt}\%$', end='')
  print(r' \\')


inpaths = [
    'runs/crafter-reward-dreamerv2.json',
    'runs/crafter-reward-rainbow.json',
    'runs/crafter-reward-ppo.json',
    'runs/crafter-reward-efficient_rainbow.json',
    'runs/crafter-reward-rnd.json',
]
legend = {
    'dreamerv2': 'DreamerV2',
    'rainbow': 'Rainbow',
    'ppo': 'PPO',
    'efficient_rainbow': 'Eff. Rainbow',
    'rnd': 'RND',
}
print_table(inpaths, legend)

inpaths = [
    'runs/crafter-noreward-rnd.json',
    'runs/crafter-noreward-random.json',
]
legend = {
    'rnd': 'RND',
    'random': 'Random',
}
print_table(inpaths, legend)
