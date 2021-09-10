import argparse

import crafter
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

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
    env.reset()
    done = False
  _, _, done, _ = env.step(action_space.sample())
  step += 1
  bar.update(1)
