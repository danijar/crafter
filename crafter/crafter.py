import pathlib

import gym
import imageio
import numpy as np
import opensimplex
import skimage.transform


TEXTURES = {
    'water': 'assets/water.png',
    'grass': 'assets/grass.png',
    'stone': 'assets/stone.png',
    'path': 'assets/path.png',
    'sand': 'assets/sand.png',
    'tree': 'assets/tree.png',
    'coal': 'assets/coal.png',
    'iron': 'assets/iron.png',
    'diamond': 'assets/diamond.png',
    'lava': 'assets/lava.png',
    'table': 'assets/table.png',
    'furnace': 'assets/furnace.png',
    'player-left': 'assets/player-left.png',
    'player-right': 'assets/player-right.png',
    'player-up': 'assets/player-up.png',
    'player-down': 'assets/player-down.png',
    'cow': 'assets/cow.png',
    'zombie': 'assets/zombie.png',
}

MATERIAL_NAMES = {
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'path',
    5: 'sand',
    6: 'tree',
    7: 'lava',
    8: 'coal',
    9: 'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
}

MATERIAL_IDS = {
    name: id_ for id_, name in MATERIAL_NAMES.items()
}

WALKABLE = {
    MATERIAL_IDS['grass'],
    MATERIAL_IDS['path'],
    MATERIAL_IDS['sand'],
}


class Player:

  def __init__(self, pos):
    self.pos = pos
    self.face = (0, 1)
    self.health = 3
    self.inventory = {
        'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0,
        'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0,
        # 'wood_sword': 0, 'stone_sword': 0, 'iron_sword': 0,
    }

  @property
  def texture(self):
    return {
        (-1, 0): 'player-left',
        (+1, 0): 'player-right',
        (0, -1): 'player-up',
        (0, +1): 'player-down',
    }[self.face]

  def update(self, terrain, objects, action):
    if action == 0:
      return  # noop
    if 1 <= action <= 4:
      # left, right, up, down
      direction = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
      self.face = direction[action - 1]
      target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
      if _is_free(target, terrain, objects):
        self.pos = target
      return
    target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
    empty = MATERIAL_NAMES[terrain[target]] in ('grass', 'sand', 'path')
    water = MATERIAL_NAMES[terrain[target]] in ('water',)
    if action == 5:  # grab
      pickaxe = max(
          1 if self.inventory['wood_pickaxe'] else 0,
          2 if self.inventory['stone_pickaxe'] else 0,
          3 if self.inventory['iron_pickaxe'] else 0)
      if terrain[target] == MATERIAL_IDS['tree']:
        terrain[target] = MATERIAL_IDS['grass']
        self.inventory['wood'] += 1
      elif terrain[target] == MATERIAL_IDS['stone'] and pickaxe > 0:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['stone'] += 1
      elif terrain[target] == MATERIAL_IDS['coal'] and pickaxe > 0:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['coal'] += 1
      elif terrain[target] == MATERIAL_IDS['iron'] and pickaxe > 1:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['iron'] += 1
      elif terrain[target] == MATERIAL_IDS['diamond'] and pickaxe > 2:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['diamond'] += 1
      return
    if action == 6:  # attack
      for obj in objects:
        if obj.pos == target and hasattr(obj, 'health'):
          obj.health -= 1
        if isinstance(obj, Cow) and obj.health <= 0:
          self.health = min(self.health + 1, 3)  # food
      return
    if action == 7:  # place stone
      if self.inventory['stone'] > 0 and (empty or water):
        terrain[target] = MATERIAL_IDS['stone']
        self.inventory['stone'] -= 1
      return
    if action == 8:  # place table
      if self.inventory['wood'] > 0 and empty:
        terrain[target] = MATERIAL_IDS['table']
        self.inventory['wood'] -= 1
      return
    if action == 9:  # place furnace
      if self.inventory['stone'] > 0 and empty:
        terrain[target] = MATERIAL_IDS['furnace']
        self.inventory['stone'] -= 1
      return
    nearby = terrain[
        self.pos[0] - 2: self.pos[0] + 2,
        self.pos[1] - 2: self.pos[1] + 2]
    table = (nearby == MATERIAL_IDS['table']).any()
    furnace = (nearby == MATERIAL_IDS['furnace']).any()
    if action == 10:  # make wood pickaxe
      if self.inventory['wood'] > 0 and table:
        self.inventory['wood'] -= 1
        self.inventory['wood_pickaxe'] += 1
    if action == 11:  # make stone pickaxe
      wood = self.inventory['wood']
      stone = self.inventory['stone']
      if wood > 0 and stone > 0 and table:
        self.inventory['wood'] -= 1
        self.inventory['stone'] -= 1
        self.inventory['stone_pickaxe'] += 1
    if action == 12:  # make iron pickaxe
      wood = self.inventory['wood']
      iron = self.inventory['iron']
      if wood and iron and furnace:
        self.inventory['wood'] -= 1
        self.inventory['stone'] -= 1
        self.inventory['iron_pickaxe'] += 1


class Cow:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random

  @property
  def texture(self):
    return 'cow'

  def update(self, terrain, objects, action):
    if self.health <= 0:
      del objects[objects.index(self)]
    x = self.pos[0] + self._random.randint(-1, 2)
    y = self.pos[1] + self._random.randint(-1, 2)
    if _is_free((x, y), terrain, objects):
      self.pos = (x, y)


class Zombie:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random
    self._near = False

  @property
  def texture(self):
    return 'zombie'

  def update(self, terrain, objects, action):
    if self.health <= 0:
      del objects[objects.index(self)]
    player = [obj for obj in objects if isinstance(obj, Player)][0]
    dist = np.sqrt(
        (self.pos[0] - player.pos[0]) ** 2 +
        (self.pos[1] - player.pos[1]) ** 2)
    if dist <= 1:
      if self._near and self._random.uniform() > 0.5:
        player.health -= 1
      self._near = True
    else:
      self._near = False
    if dist <= 4:
      if abs(self.pos[0] - player.pos[0]) > abs(self.pos[1] - player.pos[1]):
        direction = (-np.sign(self.pos[0] - player.pos[0]), 0)
      else:
        direction = (0, -np.sign(self.pos[1] - player.pos[1]))
    else:
      if self._random.uniform() > 0.5:
        direction = (0, self._random.randint(-1, 2))
      else:
        direction = (self._random.randint(-1, 2), 0)
    x = self.pos[0] + direction[0]
    y = self.pos[1] + direction[1]
    if _is_free((x, y), terrain, objects):
      self.pos = (x, y)


class Env:

  def __init__(self, area=(64, 64), view=5, size=64, seed=2):
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
    # noop, left, right, up, down, grab, attack, place stone, place table,
    # place furnace, make wood pickaxe, make stone pickaxe, make iron pickaxe
    return gym.spaces.Discrete(13)

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
    center = self._area[0] // 2, self._area[1] // 2
    self._simplex = opensimplex.OpenSimplex(
        seed=hash((self._seed, self._episode)))
    self._random = np.random.RandomState(
        seed=np.uint32(hash((self._seed, self._episode))))
    simplex = self._noise
    uniform = self._random.uniform

    for x in range(self._area[0]):
      for y in range(self._area[1]):
        start = 4 - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        start += 2 * simplex(x, y, 8, 3)
        start = 1 / (1 + np.exp(-start))
        mountain = simplex(x, y, 0, {1: 15, 0.3: 5}) - 3 * start
        if start > 0.5:
          self._terrain[x, y] = MATERIAL_IDS['grass']
        elif mountain > 0.15:
          if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(x, y, 1, 8) > 0 and uniform() > 0.8:
            self._terrain[x, y] = MATERIAL_IDS['coal']
          elif simplex(x, y, 2, 6) > 0.3 and uniform() > 0.6:
            self._terrain[x, y] = MATERIAL_IDS['iron']
          elif mountain > 0.25 and uniform() > 0.99:
            self._terrain[x, y] = MATERIAL_IDS['diamond']
          elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.4:
            self._terrain[x, y] = MATERIAL_IDS['lava']
          else:
            self._terrain[x, y] = MATERIAL_IDS['stone']
        elif 0.25 < simplex(x, y, 3, 15) <= 0.35 and simplex(x, y, 4, 9) > -0.2:
          self._terrain[x, y] = MATERIAL_IDS['sand']
        elif simplex(x, y, 3, 15) > 0.3:
          self._terrain[x, y] = MATERIAL_IDS['water']
        else:  # grass
          if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
            self._terrain[x, y] = MATERIAL_IDS['tree']
          else:
            self._terrain[x, y] = MATERIAL_IDS['grass']

    self._player = Player(center)
    self._objects = [self._player]
    for x in range(self._area[0]):
      for y in range(self._area[1]):
        if self._terrain[x, y] in WALKABLE:
          if self._terrain[x, y] == MATERIAL_IDS['grass'] and uniform() > 0.99:
            self._objects.append(Cow((x, y), self._random))
          elif uniform() > 0.993:
            self._objects.append(Zombie((x, y), self._random))

    return self._obs()

  def step(self, action):
    for obj in self._objects:
      obj.update(self._terrain, self._objects, action)
    obs = self._obs()
    reward = self.reward()
    done = False
    info = {}
    return obs, reward, done, info

  def reward(self):
    return 0

  def render(self):
    canvas = np.zeros((self._size, self._size, 3), np.uint8)
    for i in range(2 * self._view + 2):
      for j in range(2 * self._view + 2):
        x = self._player.pos[0] + i - self._view
        y = self._player.pos[1] + j - self._view
        if not (0 <= x < self._area[0] and 0 <= y < self._area[1]):
          continue
        texture = self._textures[MATERIAL_NAMES[self._terrain[x, y]]]
        self._draw(canvas, (x, y), texture)
    for obj in self._objects:
      texture = self._textures[obj.texture]
      self._draw(canvas, obj.pos, texture)
    used = self._grid * (2 * self._view + 1)
    # if used != self._size:
    #   canvas = skimage.transform.resize(
    #       canvas[:used, :used], (self._size, self._size),
    #       order=0, anti_aliasing=False,
    #       preserve_range=True).astype(np.uint8)
    return canvas

  def _obs(self):
    obs = {
        'image': self.render(),
        'player': self._player.pos,
    }
    return obs

  def _draw(self, canvas, pos, texture):
    left = self._player.pos[0] - self._view
    top = self._player.pos[1] - self._view
    x = self._grid * (pos[0] - left)
    y = self._grid * (pos[1] - top)
    w = texture.shape[0]
    h = texture.shape[1]
    if not (0 <= x and x + w <= canvas.shape[0]): return
    if not (0 <= y and y + h <= canvas.shape[1]): return
    if texture.shape[-1] == 4:
      alpha = texture[..., 3:].astype(np.float32) / 255
      texture = texture[..., :3].astype(np.float32) / 255
      current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
      blended = alpha * texture + (1 - alpha) * current
      result = (255 * blended).astype(np.uint8)
    else:
      result = texture
    canvas[x: x + w, y: y + h] = result


  def _load_textures(self):
    textures = {}
    resolution = (self._size // 2 * self._view + 1)
    for name, filename in TEXTURES.items():
      filename = pathlib.Path(__file__).parent / filename
      image = imageio.imread(filename)
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      image = skimage.transform.resize(
          image, (self._grid, self._grid),
          order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
      textures[name] = image
    return textures


def _is_free(pos, terrain, objects):
  if not (0 <= pos[0] < terrain.shape[0]): return False
  if not (0 <= pos[1] < terrain.shape[1]): return False
  if terrain[pos[0], pos[1]] not in WALKABLE: return False
  if any(obj.pos == pos for obj in objects): return False
  return True


def test_map():
  env = Env(area=(64, 64), view=31, size=1024, seed=0)
  images = []
  for _ in range(4):
    images.append(env.reset()['image'])
  grid = np.concatenate([
      np.concatenate([images[0], images[1]], 1),
      np.concatenate([images[2], images[3]], 1),
  ], 0)
  imageio.imsave('map.png', grid.transpose((1, 0, 2)))
  print('Saved map.png')


def test_episode():
  import time
  env = Env(area=(64, 64), view=4, size=64, seed=0)
  start = time.time()
  env.reset()
  print(f'Reset time: {time.time()-start:.2f}s')
  frames = []
  random = np.random.RandomState(0)
  start = time.time()
  for index in range(100):
    action = random.randint(0, env.action_space.n)
    obs, _, _, _ = env.step(action)
    frames.append(obs['image'])
  duration = time.time() - start
  print(f'Step time: {duration:.2f}s ({int(100/duration)} FPS)')
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


def test_keyboard(size=500, recording=True):
  import pygame
  pygame.init()
  env = Env(area=(64, 64), view=4, size=size, seed=0)
  env.reset()
  noop = 0
  keymap = {
      pygame.K_a: 1,       # left
      pygame.K_d: 2,       # right
      pygame.K_w: 3,       # up
      pygame.K_s: 4,       # down
      pygame.K_SPACE: 5,   # grab
      pygame.K_RETURN: 6,  # attack
      pygame.K_1: 7,       # place stone
      pygame.K_2: 8,       # place table
      pygame.K_3: 9,       # place furnace
      pygame.K_4: 10,      # make wood pickaxe
      pygame.K_5: 11,      # make stone pickaxe
      pygame.K_6: 12,      # make iron pickaxe
  }
  if recording:
    frames = []
  screen = pygame.display.set_mode([size, size])
  running = True
  clock = pygame.time.Clock()
  while running:
    action = None
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        action = noop
    obs, _, _, _ = env.step(action)
    if action > 4:
      print('Inventory:', ', '.join(sorted(
          f'{k}: {v}' for k, v in env._player.inventory.items())))
      print('Health:', env._player.health)
    if recording:
      frames.append(obs['image'].transpose((1, 0, 2)))
    surface = pygame.surfarray.make_surface(obs['image'])
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(3)  # fps
  pygame.quit()
  if recording:
    imageio.mimsave('recording.mp4', frames)


if __name__ == '__main__':
  # test_map()
  # test_episode()
  test_keyboard()
