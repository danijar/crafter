import numpy as np

import common


def print_reward(inpaths, legend, budget=1e6, last=1e5, sort=False):
  runs = common.load_runs(inpaths, budget)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  seeds = sorted({x['seed'] for x in runs})
  rewards = np.empty((len(legend), len(seeds)))
  rewards[:] = np.nan
  for i, (method, label) in enumerate(legend.items()):
    relevant = [run for run in runs if run['method'] == method]
    if not relevant:
      print(f'No runs found for method {method}.')
    for run in relevant:
      j = seeds.index(run['seed'])
      xs = np.array(run['xs'])
      ys = np.array(run['reward'])
      rewards[i][j] = ys[-(xs >= (xs.max() - last)).sum()]
  means = np.nanmean(rewards, -1)
  stds = np.nanstd(rewards, -1)
  print('')
  print(r'\textbf{Method} & \textbf{Reward} \\')
  print('')
  for method, mean, std in zip(legend.values(), means, stds):
    mean = f'{mean:.1f}'
    mean = (r'\o' if len(mean) < 4 else ' ') + mean
    print(rf'{method:<25} & ${mean} \pm {std:4.1f}$ \\')
  print('')


inpaths = [
    'scores/crafter_reward-dreamerv2.json',
    'scores/crafter_reward-ppo.json',
    'scores/crafter_reward-rainbow.json',
    'scores/crafter_noreward-unsup_plan2explore.json',
    'scores/crafter_noreward-unsup_rnd.json',
    'scores/crafter_noreward-random.json',
    'scores/crafter_reward-human.json',
]
legend = {
    'human': 'Human Experts',
    'dreamerv2': 'DreamerV2',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',
    'unsup_plan2explore': 'Plan2Explore (Unsup)',
    'unsup_rnd': 'RND (Unsup)',
    'random': 'Random',
}
print_reward(inpaths, legend)
