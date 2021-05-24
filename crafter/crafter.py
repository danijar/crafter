import numpy as np
import opensimplex

from . import engine


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

MATERIALS = [
    'water', 'grass', 'stone', 'path', 'sand', 'tree', 'lava', 'coal', 'iron',
    'diamond', 'table', 'furnace',
]

WALKABLE = ['grass', 'path', 'sand']


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
      elif _is_free(target, terrain, objects, ['lava']):
        objects.move(self, target)
        self.health = 0
      return
    target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
    area = terrain.area
    if (0 <= target[0] < area[0]) and (0 <= target[1] < area[1]):
      material = terrain[target]
    else:
      material = -1
      material = 'out_of_bounds'
    empty = material in ('grass', 'sand', 'path')
    water = material in ('water',)
    lava = material in ('lava',)
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
      if material == 'tree':
        terrain[target] = 'grass'
        self.inventory['wood'] += 1
        self.achievements.add('collect_wood')
      elif material == 'stone' and pickaxe > 0:
        terrain[target] = 'path'
        self.inventory['stone'] += 1
        self.achievements.add('collect_stone')
      elif material == 'coal' and pickaxe > 0:
        terrain[target] = 'path'
        self.inventory['coal'] += 1
        self.achievements.add('collect_coal')
      elif material == 'iron' and pickaxe > 1:
        terrain[target] = 'path'
        self.inventory['iron'] += 1
        self.achievements.add('collect_iron')
      elif material == 'diamond' and pickaxe > 2:
        terrain[target] = 'path'
        self.inventory['diamond'] += 1
        self.achievements.add('collect_diamond')
      return
    if action == 6:  # place stone
      if self.inventory['stone'] > 0 and (empty or water or lava):
        terrain[target] = 'stone'
        self.inventory['stone'] -= 1
        self.achievements.add('place_stone')
      return
    if action == 7:  # place table
      if self.inventory['wood'] > 0 and empty:
        terrain[target] = 'table'
        self.inventory['wood'] -= 1
        self.achievements.add('place_table')
      return
    if action == 8:  # place furnace
      if self.inventory['stone'] > 0 and empty:
        terrain[target] = 'furnace'
        self.inventory['stone'] -= 1
        self.achievements.add('place_furnace')
      return
    nearby = terrain.nearby(self.pos, 2)
    table = ('table' in nearby)
    furnace = ('furnace' in nearby)
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
      self, area=(64, 64), view=(4, 4), size=(64, 64), length=10000, health=5,
      seed=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    self._area = area
    self._size = size
    self._length = length
    self._health = health
    self._seed = seed
    self._episode = 0
    self._grid = size // (2 * view + 1)
    self._border = (size - self._grid * (2 * view + 1)) // 2
    self._textures = engine.Textures(TEXTURES, self._grid)
    self._terrain = engine.Terrain(MATERIALS, area)
    self._objects = engine.Objects(area)
    self._view = engine.LocalView(
        self._terrain, self._objects, self._textures, self._grid, view)
    self._step = None
    self._random = None
    self._player = None
    self._simplex = None
    self._achievements = None
    self._last_health = None

  @property
  def observation_space(self):
    shape = (self._size, self._size, 3)
    spaces = {'image': engine.BoxSpace(0, 255, shape, np.uint8)}
    inventory = Player((0, 0), self._health).inventory
    for key in list(inventory.keys()) + ['health']:
      spaces[key] = engine.BoxSpace(0, 255, (), np.uint8)
    return engine.DictSpace(spaces)

  @property
  def action_space(self):
    return engine.DiscreteSpace(12)

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
    self._terrain.reset()
    self._objects.reset()
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
          self._terrain[x, y] = 'grass'
        elif mountain > 0.15:
          if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
            self._terrain[x, y] = 'path'
          elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
            self._terrain[x, y] = 'path'
          elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
            self._terrain[x, y] = 'path'
          elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
            self._terrain[x, y] = 'coal'
          elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
            self._terrain[x, y] = 'iron'
          elif mountain > 0.18 and uniform() > 0.995:
            self._terrain[x, y] = 'diamond'
          elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.4:
            self._terrain[x, y] = 'lava'
          else:
            self._terrain[x, y] = 'stone'
        elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
          self._terrain[x, y] = 'sand'
        elif 0.3 < water:
          self._terrain[x, y] = 'water'
        else:  # grassland
          if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
            self._terrain[x, y] = 'tree'
          else:
            self._terrain[x, y] = 'grass'
    self._player = Player(center, self._health)
    self._last_health = self._health
    self._achievements = self._player.achievements.copy()
    self._objects.add(self._player)
    for x in range(self._area[0]):
      for y in range(self._area[1]):
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        if self._terrain[x, y] in WALKABLE:
          grass = (self._terrain[x, y] == 'grass')
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
    canvas = np.zeros(tuple(self._size) + (3,), np.uint8) + 127
    image = self._view(self._player)
    (x, y), (w, h) = self._border, image.shape[:2]
    canvas[x: x + w, y: y + h] = image
    return canvas

  def _obs(self):
    obs = {'image': self.render()}
    obs['health'] = _uint8(self._player.health)
    obs.update({k: _uint8(v) for k, v in self._player.inventory.items()})
    return obs


def _is_free(pos, terrain, objects, valid=WALKABLE):
  if not (0 <= pos[0] < terrain.area[0]):
    return False
  if not (0 <= pos[1] < terrain.area[1]):
    return False
  if terrain[pos] not in valid:
    return False
  if not objects.free(pos):
    return False
  return True


def _random_direction(random):
  if random.uniform() > 0.5:
    return (0, random.randint(-1, 2))
  else:
    return (random.randint(-1, 2), 0)


def _uint8(value):
  return np.array(max(0, min(value, 255)), dtype=np.uint8)
