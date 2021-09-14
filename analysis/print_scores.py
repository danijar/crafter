import numpy as np

import common


def print_scores(inpaths, legend, budget=1e6, sort=False):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  scores = common.compute_scores(percents)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  means = np.nanmean(scores, -1)
  stds = np.nanstd(scores, -1)

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
print_scores(inpaths, legend)
