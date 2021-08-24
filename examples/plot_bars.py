import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt


def plot_bars(indir, legend, colors, budget=5e6):
  runs = []
  print('Loading runs:')
  for filename in pathlib.Path(indir).glob('**/*.json'):
    loaded = json.loads(filename.read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      print(f'- {run["method"]} seed {run["seed"]}', flush=True)
      if run['xs'][-1] < 0.99 * budget:
        print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))
  tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))

  scores = np.empty((len(methods), len(seeds), len(tasks)))
  scores[:] = np.nan
  for run in runs:
    episodes = (np.array(run['xs']) <= budget).sum()
    i = methods.index(run['method'])
    j = seeds.index(run['seed'])
    for key, values in run.items():
      if key in tasks:
        k = tasks.index(key)
        percent = 100 * (np.array(values[:episodes]) >= 1).mean()
        scores[i, j, k] = percent

  first = list(legend.keys())[0]
  order = np.argsort(-np.nanmean(scores[methods.index(first)], 0), -1)
  scores = scores[:, :, order]
  tasks = np.array(tasks)[order].tolist()

  fig, ax = plt.subplots(figsize=(7, 3))
  centers = np.arange(len(tasks))
  width = 0.7
  for index, (method, label) in enumerate(legend.items()):
    heights = np.nanmean(scores[methods.index(method)], 0)
    pos = centers + width * (0.5 / len(methods) + index / len(methods) - 0.5)
    color = colors[index]
    ax.bar(pos, heights, width / len(methods), label=label, color=color)
  ax.legend(loc='upper right', frameon=False, borderpad=0, borderaxespad=0)

  names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(axis='x', which='both', length=0)
  ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
  ax.set_xticks(centers + 0.0)
  ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')

  ax.set_ylim(0, 100)
  ax.set_ylabel('Success Rate (%)')
  yticks = [0, 20, 40, 60, 80, 100]
  ax.set_yticks(yticks)
  fig.tight_layout()
  return fig


filename = pathlib.Path('results/bars.pdf')
filename.parent.mkdir(exist_ok=True, parents=True)
legend = {'dreamerv2': 'DreamerV2', 'ppo': 'PPO', 'random': 'Random'}
colors = ['#377eb8', '#4daf4a', '#a65628', '#e41a1c', '#984ea3']
plot_bars('runs', legend, colors).savefig(filename)
print(f'Saved {filename}')
