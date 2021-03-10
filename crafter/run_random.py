import time

import imageio
import numpy as np

import crafter

env = crafter.Env(area=(64, 64), view=4, size=64, seed=0)
start = time.time()
env.reset()
print(f'Reset time: {1000*(time.time()-start):.2f}ms')
frames = []
random = np.random.RandomState(0)
start = time.time()
for index in range(100):
  action = random.randint(0, env.action_space.n)
  obs, _, _, _ = env.step(action)
  frames.append(obs['image'])
duration = time.time() - start
print(f'Step time: {1000*duration/100:.2f}ms ({int(100/duration)} FPS)')
cols = int(np.sqrt(len(frames)))
rows = int(np.ceil(len(frames) / cols))
grid = np.concatenate([
    np.concatenate([frames[row * cols + col] for col in range(cols)], 1)
    for row in range(rows)
], 0)
imageio.imsave('episode.png', grid)
print('Saved episode.png')
imageio.mimsave('episode.mp4', frames)
print('Saved episode.mp4')

