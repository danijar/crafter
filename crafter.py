import gym
import imageio
import numpy as np
import opensimplex
import skimage.transform


TEXTURES = {
    'water': 'assets/water.png',
    'grass': 'assets/grass.png',
    'stone': 'assets/stone.png',
    'tree': 'assets/tree.png',
    'coal': 'assets/coal.png',
    'iron': 'assets/iron.png',
    'diamond': 'assets/diamond.png',
    'player': 'assets/player.png',
    'zombie': 'assets/zombie.png',
}

MATERIAL_NAMES = {
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'tree',
    5: 'coal',
    6: 'iron',
    7: 'diamond',
}

MATERIAL_IDS = {
    name: id_ for id_, name in MATERIAL_NAMES.items()
}


class Player:

  def __init__(self, pos):
    self.pos = pos

  def update(self, terrain, objects, action):
    if 0 <= action <= 3:
      pos = [
          (self.pos[0] - 1, self.pos[1]),  # left
          (self.pos[0] + 1, self.pos[1]),  # right
          (self.pos[0], self.pos[1] - 1),  # up
          (self.pos[0], self.pos[1] + 1),  # down
          ][action]
      if _is_free(pos, terrain, objects):
        self.pos = pos


class Zombie:

  def __init__(self, pos, random):
    self.pos = pos
    self._random = random

  def update(self, terrain, objects, action):
    x = self.pos[0] + self._random.randint(-1, 2)
    y = self.pos[1] + self._random.randint(-1, 2)
    if _is_free((x, y), terrain, objects):
      self.pos = (x, y)


class Env:

  def __init__(self, area=(64, 64), view=31, size=1024, seed=2):
    self._area = area
    self._view = view
    self._size = size
    self._seed = seed
    self._episode = 0
    self._grid = self._size // (2 * self._view + 1)
    self._textures = self._load_textures()
    self._terrain = np.zeros(area, np.uint8)
    self._random = None
    self._player = None
    self._objects = None
    self._simplex = None

  @property
  def observation_space(self):
    image = gym.spaces.Box(0, 255, (self._size, self._size, 3), dtype=np.uint8),
    coord = gym.spaces.Box([0, 0], self._area, dtype=np.int32),
    spaces = {'image': image, 'coord': coord}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return gym.spaces.Discrete(4)  # left, right, up, down

  def _noise(self, x, y, z, sizes):
    if not isinstance(sizes, dict):
      sizes = {1: sizes}
    value = 0
    for weight, size in sizes.items():
      value += weight * self._simplex.noise3d(x / size, y / size, z)
    return value / sum(sizes.keys())

  def reset(self):
    self._episode += 1
    self._terrain[:] = 0
    self._simplex = opensimplex.OpenSimplex(
        seed=hash((self._seed, self._episode)))
    self._random = np.random.RandomState(
        seed=np.uint32(hash((self._seed, self._episode))))
    simplex = self._noise
    uniform = self._random.uniform
    for x in range(self._area[0]):
      for y in range(self._area[1]):
        if simplex(x, y, 0, {1: 15, 0.3: 5}) > 0.15:
          if simplex(x, y, 1, 8) > 0.15 and uniform() > 0.8:
            self._terrain[x, y] = MATERIAL_IDS['coal']
          elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.6:
            self._terrain[x, y] = MATERIAL_IDS['iron']
          elif 0.25 < simplex(x, y, 0, {1: 15, 0.3: 5}) < 0.5 and uniform() > 0.99:
            self._terrain[x, y] = MATERIAL_IDS['diamond']
          else:
            self._terrain[x, y] = MATERIAL_IDS['stone']
        elif simplex(x, y, 3, {1: 15}) > 0.3:
          self._terrain[x, y] = MATERIAL_IDS['water']
        else:  # grass
          if simplex(x, y, 3, 7) > 0 and uniform() > 0.8:
            self._terrain[x, y] = MATERIAL_IDS['tree']
          else:
            self._terrain[x, y] = MATERIAL_IDS['grass']
    self._player = Player((self._area[0] // 2, self._area[1] // 2))
    self._objects = []
    return self._obs()

  def step(self, action):
    for obj in self._objects:
      obj.update(self._objects, action)
    obs = self._obs()
    reward = self.reward()
    done = False
    info = {}
    return obs, reward, done, info

  def reward(self):
    return 0

  def render(self):
    image = np.zeros((self._size, self._size, 3), np.uint8)
    for i in range(2 * self._view + 1):
      for j in range(2 * self._view + 1):
        x = self._player.pos[0] + i - self._view
        y = self._player.pos[0] + j - self._view
        if not (0 <= x < self._area[0] and 0 <= y < self._area[1]):
          continue
        name = MATERIAL_NAMES[self._terrain[x, y]]
        texture = self._textures[name]
        image[i * self._grid: (i + 1) * self._grid, j * self._grid: (j + 1) * self._grid] = texture
    # for obj in self._objects:
    #   visible = _view_distance(obj.pos, self._player.pos) <= self._view
    #   if partial and not visible:
    #     continue
    #   self._draw_pos(image, obj.pos, obj.color)
    return image

  def _obs(self):
    obs = {
        'image': self.render(),
        'player': self._player.pos,
    }
    return obs

  # def _random_pos(self, blocked):
  #   while True:
  #     x = self._random.randint(0, self._grid[0])
  #     y = self._random.randint(0, self._grid[1])
  #     if (x, y) not in blocked:
  #       return (x, y)

  def _draw_pos(self, canvas, pos, texture):
    if pos[0] < 0 or pos[0] >= self._area[0]:
      return
    if pos[1] < 0 or pos[1] >= self._area[1]:
      return
    w = canvas.shape[0] // self._area[0]
    h = canvas.shape[1] // self._area[1]
    x = w * pos[0]
    y = h * pos[1]
    canvas[x: x + w, y: y + h] = texture


  def _load_textures(self):
    textures = {}
    resolution = (self._size // 2 * self._view + 1)
    for name, filename in TEXTURES.items():
      image = imageio.imread(filename)
      image = skimage.transform.resize(
          image, (self._grid, self._grid),
          order=0,
          anti_aliasing=False,
          preserve_range=True)[:, :, :3]
      textures[name] = image
    return textures


def _is_free(pos, terrain, objects):
  if pos[0] < 0 or pos[0] >= terrain[0]:
    return False
  if pos[1] < 0 or pos[1] >= terrain[1]:
    return False
  if any(obj.pos == pos for obj in objects):
    return False
  return True


def _view_distance(lhs, rhs):
  return max(abs(l - r) for l, r in zip(lhs, rhs))


def test_initial():
  import matplotlib.pyplot as plt
  env = Env(area=(64, 64), view=31, size=1024, seed=0)
  images = []
  for _ in range(4):
    images.append(env.reset()['image'])
  grid = np.concatenate([
      np.concatenate([images[0], images[1]], 1),
      np.concatenate([images[2], images[3]], 1),
  ], 0)
  imageio.imsave('initial.png', grid)
  print('Saved initial.png')


def test_episode():
  import matplotlib.pyplot as plt
  env = Env()
  env.reset()
  frames = []
  rewards = []
  random = np.random.RandomState(0)
  for index in range(100):
    action = random.randint(0, env.action_space.n)
    _, reward, _, _ = env.step(action)
    image = env.render()
    frames.append(image)
    rewards.append(reward)
  cols = int(np.sqrt(len(frames)))
  rows = int(np.ceil(len(frames) / cols))
  fig, axs = plt.subplots(cols, rows, figsize=(rows, cols))
  axs = axs.flatten()
  for t, (ax, frame, r) in enumerate(zip(axs, frames, rewards)):
    ax.set_title(f'r={r:.2f}')
    ax.imshow(frame)
    ax.axis('off')
  fig.tight_layout()
  fig.savefig('episode.png')
  print('Saved episode.png')


if __name__ == '__main__':
  test_initial()
