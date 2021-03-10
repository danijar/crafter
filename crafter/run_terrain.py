import imageio

import crafter

env = crafter.Env(area=(64, 64), view=31, size=1024, seed=0)
images = []
for _ in range(4):
  images.append(env.reset()['image'])
grid = np.concatenate([
    np.concatenate([images[0], images[1]], 1),
    np.concatenate([images[2], images[3]], 1),
], 0)
imageio.imsave('map.png', grid.transpose((1, 0, 2)))
print('Saved map.png')

