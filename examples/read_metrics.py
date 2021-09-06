import collections
import json
import pathlib

import numpy as np


def read_stats(indir, outdir, task, method, budget=int(1e6)):
  indir = pathlib.Path(indir)
  outdir = pathlib.Path(outdir)
  runs = []
  print(f'Loading {indir.name}...')
  filenames = sorted(list(indir.glob('**/stats.jsonl')))
  for index, filename in enumerate(filenames):
    rewards, lengths, achievements = load_stats(filename, budget)
    if sum(lengths) < budget - 1e4:
      message = f'Skipping incomplete run ({sum(lengths)} < {budget} steps): '
      message += f'{filename.relative_to(indir.parent)}'
      print(f'==> {message}')
      continue
    runs.append(dict(
        task=task,
        method=method,
        seed=str(index),
        xs=np.cumsum(lengths).tolist(),
        reward=rewards,
        length=lengths,
        **achievements,
    ))
  print_summary(runs)
  outdir.mkdir(exist_ok=True, parents=True)
  filename = (outdir / f'{task}-{method}.json')
  filename.write_text(json.dumps(runs))
  print('Wrote', filename)
  print('')


def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


def print_summary(runs):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Lengths:      {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  print('Achievements:')
  percents = {
      k: np.mean([100 * (np.array(x[k]) >= 1).mean() for x in runs])
      for k in runs[0].keys() if k.startswith('achievement_')}
  for key, percent in sorted(percents.items(), key=lambda x: -x[1]):
    name = key[len('achievement_'):]
    print(f'{name:<20}  {percent:6.2f}%')


read_stats(
    'logdir/crafter-reward-dreamerv2',
    'runs', 'crafter-reward', 'dreamerv2')
read_stats(
    'logdir/crafter-reward-rainbow',
    'runs', 'crafter-reward', 'rainbow')
read_stats(
    'logdir/crafter-reward-efficient-rainbow',
    'runs', 'crafter-reward', 'efficient_rainbow')
read_stats(
    'logdir/crafter-reward-rnd',
    'runs', 'crafter-reward', 'rnd')
read_stats(
    'logdir/crafter-reward-ppo',
    'runs', 'crafter-reward', 'ppo')
read_stats(
    'logdir/crafter-noreward-rnd',
    'runs', 'crafter-noreward', 'rnd')
read_stats(
    'logdir/crafter-noreward-random',
    'runs', 'crafter-noreward', 'random')
