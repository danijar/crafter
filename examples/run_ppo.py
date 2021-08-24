import argparse

import crafter
import stable_baselines3

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='ppo-episodes')
parser.add_argument('--steps', type=float, default=5e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_episode=True,
    save_video=False,
    include_image=False,
)

model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)
