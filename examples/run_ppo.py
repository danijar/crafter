import argparse

import crafter
import stable_baselines3

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)

model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)
