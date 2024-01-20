import argparse

import crafter
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=float, default=1e6)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

seed = args.seed

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)
action_space = env.action_space

done = True
step = 0
bar = tqdm.tqdm(total=args.steps, smoothing=0)
while step < args.steps or not done:
  if done:
    seed = hash(seed) % (2 ** 31 - 1)
    env.reset(seed)
    done = False
  _, _, terminated, truncated, _ = env.step(action_space.sample())
  done = terminated or truncated
  step += 1
  bar.update(1)
