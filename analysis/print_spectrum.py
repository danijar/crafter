import collections

import numpy as np

import common


def print_spectrum(inpaths, legend, budget=1e6, sort=False):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  scores = common.compute_scores(percents)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  scores = np.nanmean(scores, 1)
  percents = np.nanmean(percents, 1)

  if sort:
    first = next(iter(legend.keys()))
    tasks = sorted(tasks, key=lambda task: -np.nanmean(percents[first, task]))
  legend = dict(reversed(legend.items()))

  cols = ''.join(rf' & \textbf{{{k}}}' for k in legend.values())
  print(r'\newcommand{\o}{\hphantom{0}}')
  print(r'\newcommand{\b}[1]{\textbf{#1}}')
  print('')
  print(f'{"Achievement":<20}' + cols + r' \\')
  print('')
  wins = collections.defaultdict(int)
  for task in tasks:
    k = tasks.index(task)
    if task.startswith('achievement_'):
      name = task[len('achievement_'):].replace('_', ' ').title()
    else:
      name = task.replace('_', ' ').title()
    print(f'{name:<20}', end='')
    best = max(percents[methods.index(m), k] for m in legend.keys())
    for method in legend.keys():
      i = methods.index(method)
      value = percents[i][k]
      winner = value >= 0.95 * best and value > 0
      fmt = rf'{value:.1f}\%'
      fmt = (r'\o' if len(fmt) < 6 else ' ') + fmt
      fmt = rf'\b{{{fmt}}}' if winner else f'   {fmt} '
      if winner:
        wins[method] += 1
      print(rf' & ${fmt}$', end='')
    print(r' \\')
  print('')

  print(f'{"Score":<20}', end='')
  best = max(scores[methods.index(m)] for m in legend.keys())
  for method in legend.keys():
    value = scores[methods.index(method)]
    bold = value >= 0.95 * best and value > 0
    fmt = rf'{value:.1f}\%'
    fmt = (r'\o' if len(fmt) < 6 else ' ') + fmt
    fmt = rf'\b{{{fmt}}}' if bold else f'   {fmt} '
    print(rf' & ${fmt}$', end='')
  print(r' \\')


inpaths = [
    'scores/crafter_reward-human.json',
]
legend = {
    'human': 'Human Experts',
}
print_spectrum(inpaths, legend)

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
print_spectrum(inpaths, legend)


inpaths = [
    'scores/crafter_noreward-unsup_plan2explore.json',
    'scores/crafter_noreward-unsup_rnd.json',
    'scores/crafter_noreward-random.json',
]
legend = {
    'unsup_plan2explore': 'Plan2Explore',
    'unsup_rnd': 'RND',
    'random': 'Random',
}
print_spectrum(inpaths, legend)
