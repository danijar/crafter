import collections
import pathlib

import imageio
import numpy as np
import opensimplex
from PIL import Image


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


DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
DictSpace = collections.namedtuple('DictSpace', 'spaces')


class Objects:

  def __init__(self, area):
    self._map = np.zeros(area, np.uint32)
    self._objects = [None]

  def __iter__(self):
    yield from (obj for obj in self._objects if obj)

  def add(self, obj):
    assert hasattr(obj, 'pos')
    assert self.free(obj.pos)
    self._map[obj.pos] = len(self._objects)
    self._objects.append(obj)

  def remove(self, obj):
    self._objects[self._map[obj.pos]] = None
    self._map[obj.pos] = 0

  def move(self, obj, pos):
    assert self.free(pos)
    self._map[pos] = self._map[obj.pos]
    self._map[obj.pos] = 0
    obj.pos = pos

  def free(self, pos):
    return self.at(pos) is None

  def at(self, pos):
    if not (0 <= pos[0] < self._map.shape[0]): return False
    if not (0 <= pos[1] < self._map.shape[1]): return False
    return self._objects[self._map[pos]]


class Player:

  def __init__(self, pos, health):
    self.pos = pos
    self.face = (0, 1)
    self.health = health
    self.inventory = {
        'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0,
        'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0,
    }
    self.achievements = set()
    self._max_health = health
    self._hunger = 0

  @property
  def texture(self):
    return {
        (-1, 0): 'player-left',
        (+1, 0): 'player-right',
        (0, -1): 'player-up',
        (0, +1): 'player-down',
    }[self.face]

  def update(self, terrain, objects, player, action):
    self._hunger += 1
    if self._hunger > 100:
      self.health -= 1
      self._hunger = 0
    if action == 0:
      return  # noop
    if 1 <= action <= 4:
      # left, right, up, down
      direction = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
      self.face = direction[action - 1]
      target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
      if _is_free(target, terrain, objects):
        objects.move(self, target)
      elif _is_free(target, terrain, objects, [MATERIAL_IDS['lava']]):
        objects.move(self, target)
        self.health = 0
      return
    target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
    area = terrain.shape
    if (0 <= target[0] < area[0]) and (0 <= target[1] < area[1]):
      material = terrain[target]
      material_name = MATERIAL_NAMES[material]
    else:
      material = -1
      material_name = 'out_of_bounds'
    empty = material_name in ('grass', 'sand', 'path')
    water = material_name in ('water',)
    lava = material_name in ('lava',)
    if action == 5:  # grab or attack
      obj = objects.at(target)
      if obj:
        if isinstance(obj, Zombie):
          obj.health -= 1
          if obj.health <= 0:
            self.achievements.add('defeat_zombie')
        if isinstance(obj, Cow):
          obj.health -= 1
          if obj.health <= 0:
            self.health = min(self.health + 1, self._max_health)
            self._hunger = 0
            self.achievements.add('find_food')
        return
      pickaxe = max(
          1 if self.inventory['wood_pickaxe'] else 0,
          2 if self.inventory['stone_pickaxe'] else 0,
          3 if self.inventory['iron_pickaxe'] else 0)
      if material == MATERIAL_IDS['tree']:
        terrain[target] = MATERIAL_IDS['grass']
        self.inventory['wood'] += 1
        self.achievements.add('collect_wood')
      elif material == MATERIAL_IDS['stone'] and pickaxe > 0:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['stone'] += 1
        self.achievements.add('collect_stone')
      elif material == MATERIAL_IDS['coal'] and pickaxe > 0:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['coal'] += 1
        self.achievements.add('collect_coal')
      elif material == MATERIAL_IDS['iron'] and pickaxe > 1:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['iron'] += 1
        self.achievements.add('collect_iron')
      elif material == MATERIAL_IDS['diamond'] and pickaxe > 2:
        terrain[target] = MATERIAL_IDS['path']
        self.inventory['diamond'] += 1
        self.achievements.add('collect_diamond')
      return
    if action == 6:  # place stone
      if self.inventory['stone'] > 0 and (empty or water or lava):
        terrain[target] = MATERIAL_IDS['stone']
        self.inventory['stone'] -= 1
        self.achievements.add('place_stone')
      return
    if action == 7:  # place table
      if self.inventory['wood'] > 0 and empty:
        terrain[target] = MATERIAL_IDS['table']
        self.inventory['wood'] -= 1
        self.achievements.add('place_table')
      return
    if action == 8:  # place furnace
      if self.inventory['stone'] > 0 and empty:
        terrain[target] = MATERIAL_IDS['furnace']
        self.inventory['stone'] -= 1
        self.achievements.add('place_furnace')
      return
    nearby = terrain[
        self.pos[0] - 2: self.pos[0] + 2,
        self.pos[1] - 2: self.pos[1] + 2]
    table = (nearby == MATERIAL_IDS['table']).any()
    furnace = (nearby == MATERIAL_IDS['furnace']).any()
    if action == 9:  # make wood pickaxe
      if self.inventory['wood'] > 0 and table:
        self.inventory['wood'] -= 1
        self.inventory['wood_pickaxe'] += 1
        self.achievements.add('make_wood_pickaxe')
    if action == 10:  # make stone pickaxe
      wood = self.inventory['wood']
      stone = self.inventory['stone']
      if wood > 0 and stone > 0 and table:
        self.inventory['wood'] -= 1
        self.inventory['stone'] -= 1
        self.inventory['stone_pickaxe'] += 1
        self.achievements.add('make_stone_pickaxe')
    if action == 11:  # make iron pickaxe
      wood = self.inventory['wood']
      coal = self.inventory['coal']
      iron = self.inventory['iron']
      if wood > 0 and coal > 0 and iron > 0 and table and furnace:
        self.inventory['wood'] -= 1
        self.inventory['coal'] -= 1
        self.inventory['iron'] -= 1
        self.inventory['iron_pickaxe'] += 1
        self.achievements.add('make_iron_pickaxe')


class Cow:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random

  @property
  def texture(self):
    return 'cow'

  def update(self, terrain, objects, player, action):
    if self.health <= 0:
      objects.remove(self)
    if self._random.uniform() < 0.5:
      return
    direction = _random_direction(self._random)
    x = self.pos[0] + direction[0]
    y = self.pos[1] + direction[1]
    if _is_free((x, y), terrain, objects):
      objects.move(self, (x, y))


class Zombie:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random
    self._near = False

  @property
  def texture(self):
    return 'zombie'

  def update(self, terrain, objects, player, action):
    if self.health <= 0:
      objects.remove(self)
    dist = np.sqrt(
        (self.pos[0] - player.pos[0]) ** 2 +
        (self.pos[1] - player.pos[1]) ** 2)
    if dist <= 1:
      if self._near and self._random.uniform() > 0.7:
        player.health -= 1
      self._near = True
    else:
      self._near = False
    if dist <= 4:
      xdist = abs(self.pos[0] - player.pos[0])
      ydist = abs(self.pos[1] - player.pos[1])
      if self._random.uniform() < 0.2:
        direction = _random_direction(self._random)
      elif xdist > ydist and self._random.uniform() < 0.7:
        direction = (-np.sign(self.pos[0] - player.pos[0]), 0)
      else:
        direction = (0, -np.sign(self.pos[1] - player.pos[1]))
    else:
      direction = _random_direction(self._random)
    x = self.pos[0] + direction[0]
    y = self.pos[1] + direction[1]
    if _is_free((x, y), terrain, objects):
      objects.move(self, (x, y))


class Env:

  def __init__(
      self, area=(64, 64), view=4, size=64, length=10000, health=5,
      seed=None):
    self._area = area
    self._view = view
    self._size = size
    self._length = length
    self._health = health
    self._seed = seed
    self._episode = 0
    self._grid = self._size // (2 * self._view + 1)
    self._textures = self._load_textures()
    self._terrain = np.zeros(area, np.uint8)
    self._border = (size - self._grid * (2 * self._view + 1)) // 2
    self._step = None
    self._random = None
    self._player = None
    self._objects = None
    self._simplex = None
    self._achievements = None
    self._last_health = None

  @property
  def observation_space(self):
    shape = (self._size, self._size, 3)
    spaces = {'image': BoxSpace(0, 255, shape, np.uint8)}
    inventory = Player((0, 0), self._health).inventory
    for key in list(inventory.keys()) + ['health']:
      spaces[key] = BoxSpace(0, 255, (), np.uint8)
    return DictSpace(spaces)

  @property
  def action_space(self):
    return DiscreteSpace(12)

  @property
  def action_names(self):
    return [
        'noop', 'left', 'right', 'up', 'down', 'interact',
        'place_stone', 'place_table', 'place_furnace',
        'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
    ]

  def _noise(self, x, y, z, sizes, normalize=True):
    if not isinstance(sizes, dict):
      sizes = {sizes: 1}
    value = 0
    for size, weight in sizes.items():
      value += weight * self._simplex.noise3d(x / size, y / size, z)
    if normalize:
      value /= sum(sizes.values())
    return value

  def reset(self):
    self._step = 0
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
        water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
        water -= 2 * start
        mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
        mountain -= 4 * start + 0.3 * water
        if start > 0.5:
          self._terrain[x, y] = MATERIAL_IDS['grass']
        elif mountain > 0.15:
          if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
            self._terrain[x, y] = MATERIAL_IDS['path']
          elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
            self._terrain[x, y] = MATERIAL_IDS['coal']
          elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
            self._terrain[x, y] = MATERIAL_IDS['iron']
          elif mountain > 0.18 and uniform() > 0.995:
            self._terrain[x, y] = MATERIAL_IDS['diamond']
          elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.4:
            self._terrain[x, y] = MATERIAL_IDS['lava']
          else:
            self._terrain[x, y] = MATERIAL_IDS['stone']
        elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
          self._terrain[x, y] = MATERIAL_IDS['sand']
        elif 0.3 < water:
          self._terrain[x, y] = MATERIAL_IDS['water']
        else:  # grassland
          if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
            self._terrain[x, y] = MATERIAL_IDS['tree']
          else:
            self._terrain[x, y] = MATERIAL_IDS['grass']

    self._player = Player(center, self._health)
    self._last_health = self._health
    self._achievements = self._player.achievements.copy()
    self._objects = Objects(self._area)
    self._objects.add(self._player)
    for x in range(self._area[0]):
      for y in range(self._area[1]):
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        if self._terrain[x, y] in WALKABLE:
          grass = self._terrain[x, y] == MATERIAL_IDS['grass']
          if dist > 3 and grass and uniform() > 0.98:
            self._objects.add(Cow((x, y), self._random))
          elif dist > 6 and uniform() > 0.993:
            self._objects.add(Zombie((x, y), self._random))

    return self._obs()

  def step(self, action):
    self._step += 1
    for obj in self._objects:
      obj.update(self._terrain, self._objects, self._player, action)
    obs = self._obs()
    reward = 0.0
    if len(self._player.achievements) > len(self._achievements):
      self._achievements = self._player.achievements.copy()
      reward += 1.0
    if self._player.health < self._last_health:
      self._last_health = self._player.health
      reward -= 0.1
    elif self._player.health > self._last_health:
      self._last_health = self._player.health
      reward += 0.1
    dead = self._player.health <= 0
    over = self._length and self._step >= self._length
    done = dead or over
    info = {
        'achievements': self._achievements.copy(),
        'discount': 1 - float(dead),
    }
    return obs, reward, done, info

  def render(self):
    canvas = np.zeros((self._size, self._size, 3), np.uint8) + 127
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
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    obs = {'image': self.render()}
    obs['health'] = _uint8(self._player.health)
    obs.update({k: _uint8(v) for k, v in self._player.inventory.items()})
    # for key, value in self._player.inventory.items():
    #   obs[key] = np.clip(value, 0, 255).astype(np.uint8)
    return obs

  def _draw(self, canvas, pos, texture):
    # TODO: This function is slow.
    x = self._grid * (pos[0] + self._view - self._player.pos[0]) + self._border
    y = self._grid * (pos[1] + self._view - self._player.pos[1]) + self._border
    w, h = texture.shape[:2]
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
    for name, filename in TEXTURES.items():
      filename = pathlib.Path(__file__).parent / filename
      image = imageio.imread(filename)
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      image = np.array(Image.fromarray(image).resize(
          (self._grid, self._grid), resample=Image.NEAREST))
      textures[name] = image
    return textures


def _is_free(pos, terrain, objects, valid=WALKABLE):
  if not (0 <= pos[0] < terrain.shape[0]): return False
  if not (0 <= pos[1] < terrain.shape[1]): return False
  if terrain[pos] not in valid: return False
  if not objects.free(pos): return False
  return True


def _random_direction(random):
  if random.uniform() > 0.5:
    return (0, random.randint(-1, 2))
  else:
    return (random.randint(-1, 2), 0)


def _uint8(value):
  # return np.clip(value, 0, 255).astype(np.uint8)
  return np.array(max(0, min(value, 255)), dtype=np.uint8)
