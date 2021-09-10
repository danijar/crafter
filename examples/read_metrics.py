import collections
import json
import pathlib

import numpy as np


def read_stats(indir, outdir, task, method, budget=int(1e6), verbose=False):
  indir = pathlib.Path(indir)
  outdir = pathlib.Path(outdir)
  runs = []
  print(f'Loading {indir.name}...')
  filenames = sorted(list(indir.glob('**/stats.jsonl')))
  for index, filename in enumerate(filenames):
    if not filename.is_file():
      continue
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
  if not runs:
    print('No completed runs.\n')
    return
  print_summary(runs, verbose)
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


def print_summary(runs, verbose):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  if verbose:
    percents = {
        k: np.mean([100 * (np.array(x[k]) >= 1).mean() for x in runs])
        for k in runs[0].keys() if k.startswith('achievement_')}
    for key, percent in sorted(percents.items(), key=lambda x: -x[1]):
      name = key[len('achievement_'):].replace('_', ' ').title()
      print(f'{name:<20}  {percent:6.2f}%')


read_stats(
    'logdir/crafter_reward-dreamerv2',
    'runs', 'crafter_reward', 'dreamerv2')
read_stats(
    'logdir/crafter_reward-ppo',
    'runs', 'crafter_reward', 'ppo')
read_stats(
    'logdir/crafter_reward-rainbow',
    'runs', 'crafter_reward', 'rainbow')
read_stats(
    'logdir/crafter_noreward-unsup_rnd',
    'runs', 'crafter_noreward', 'unsup_rnd')
read_stats(
    'logdir/crafter_noreward-unsup_plan2explore',
    'runs', 'crafter_noreward', 'unsup_plan2explore')
read_stats(
    'logdir/crafter_noreward-random',
    'runs', 'crafter_noreward', 'random')
