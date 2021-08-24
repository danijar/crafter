import argparse

import crafter
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='random-episodes')
parser.add_argument('--steps', type=float, default=5e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_episode=True,
    save_video=False,
    include_image=False,
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
