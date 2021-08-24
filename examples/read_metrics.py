import collections
import json
import pathlib

import numpy as np
from tqdm import tqdm


def read_metrics(indir, method, seed):
  indir = pathlib.Path(indir)

  scores = []
  lengths = []
  achievements = collections.defaultdict(list)

  print(f'Loading {indir}...')
  for filename in tqdm(sorted(list(indir.glob('*.npz')))):
    episode = np.load(filename)
    scores.append(np.round(np.sum(episode['reward']), 1))
    lengths.append(len(episode['reward']))
    for key in episode.keys():
      if key.startswith('achievement_'):
        achievements[key].append(int(episode[key][-1]))

  print('')
  print(f'Episodes: {len(scores)}')
  print(f'Steps:    {sum(lengths)}')
  print(f'Score:    {np.mean(scores):.2f} ± {np.std(scores):.2f}')
  print(f'Length:   {np.mean(lengths):.2f} ± {np.std(lengths):.2f}')
  print('')
  print('Achievements:')
  for key, values in sorted(achievements.items(), key=lambda x: -sum(x[1])):
    name = key[len('achievement_'):].replace('_', ' ').title()
    percent = 100 * (np.array(values) >= 1.0).mean()
    print(f'{name:<20} {percent:6.2f}%')
  print('')

  run = dict(
      task='crafter',
      method=method,
      seed=seed,
      xs=np.cumsum(lengths).tolist(),
      score=scores,
      length=lengths,
      **achievements,
  )
  return run


filename = pathlib.Path('runs/crafter-ppo.json')
filename.parent.mkdir(exist_ok=True, parents=True)
filename.write_text(json.dumps([
    read_metrics('ppo-episodes/seed-0', 'ppo', '0'),
    read_metrics('ppo-episodes/seed-1', 'ppo', '1'),
    read_metrics('ppo-episodes/seed-2', 'ppo', '2'),
    read_metrics('ppo-episodes/seed-3', 'ppo', '3'),
    read_metrics('ppo-episodes/seed-4', 'ppo', '4'),
    read_metrics('ppo-episodes/seed-5', 'ppo', '5'),
    read_metrics('ppo-episodes/seed-6', 'ppo', '6'),
    read_metrics('ppo-episodes/seed-7', 'ppo', '7'),
    read_metrics('ppo-episodes/seed-8', 'ppo', '8'),
]))
