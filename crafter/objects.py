import numpy as np

from . import constants
from . import engine


class Object:

  def __init__(self, world, pos):
    self.world = world
    self.pos = np.array(pos)
    self.random = world.random
    self.health = 0

  @property
  def texture(self):
    raise 'unknown'

  @property
  def walkable(self):
    return constants.walkable

  def move(self, direction):
    direction = np.array(direction)
    target = self.pos + direction
    if self.is_free(target):
      self.world.move(self, target)
      return True
    return False

  def is_free(self, target, materials=None):
    materials = self.walkable if materials is None else materials
    material, obj = self.world[target]
    return obj is None and material in materials

  def distance(self, target):
    if hasattr(target, 'pos'):
      target = target.pos
    return np.abs(target - self.pos).sum()

  def toward(self, target, long_axis=True):
    if hasattr(target, 'pos'):
      target = target.pos
    offset = target - self.pos
    dists = np.abs(offset)
    if (dists[0] > dists[1] if long_axis else dists[0] <= dists[1]):
      return np.array((np.sign(offset[0]), 0))
    else:
      return np.array((0, np.sign(offset[1])))

  def random_dir(self):
    directions = ((-1, 0), (+1, 0), (0, -1), (0, +1))
    return directions[self.random.randint(0, len(directions))]


class Player(Object):

  def __init__(self, world, pos, health):
    super().__init__(world, pos)
    self.facing = (0, 1)
    self.health = health
    self.inventory = {item: 0 for item in constants.items}
    self.achievements = {name: 0 for name in constants.achievements}
    self._max_health = health
    self._hunger = 0

  @property
  def texture(self):
    return {
        (-1, 0): 'player-left',
        (+1, 0): 'player-right',
        (0, -1): 'player-up',
        (0, +1): 'player-down',
    }[tuple(self.facing)]

  @property
  def walkable(self):
    return constants.walkable + ['lava']

  def update(self, action):
    self._hunger += 1
    if self._hunger > 100:
      self.health -= 1
      self._hunger = 0
    target = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
    material, obj = self.world[target]
    action = constants.actions[action]
    if action == 'noop':
      pass
    elif action.startswith('move_'):
      self._move(action[len('move_'):])
    elif action == 'do' and obj:
      self._interact(obj)
    elif action == 'do':
      self._collect(target, material)
    elif action.startswith('place_'):
      self._place(action[len('place_'):], target, material)
    elif action.startswith('make_'):
      self._make(action[len('make_'):])
    for item, amount in self.inventory.items():
      self.inventory[item] = max(0, min(amount, 5))

  def _move(self, direction):
    directions = dict(left=(-1, 0), right=(+1, 0), up=(0, -1), down=(0, +1))
    self.facing = directions[direction]
    self.move(self.facing)
    if self.world[self.pos][0] == 'lava':
      self.health = 0

  def _interact(self, obj):
    damage = max([
        1,
        self.inventory['wood_sword'] and 2,
        self.inventory['stone_sword'] and 3,
        self.inventory['iron_sword'] and 5,
    ])
    if isinstance(obj, Zombie):
      obj.health -= damage
      if obj.health <= 0:
        self.achievements['defeat_zombie'] += 1
    if isinstance(obj, Skeleton):
      obj.health -= damage
      if obj.health <= 0:
        self.achievements['defeat_skeleton'] += 1
    if isinstance(obj, Cow):
      obj.health -= damage
      if obj.health <= 0:
        self.health = min(self.health + 1, self._max_health)
        self._hunger = 0
        self.achievements['find_food'] += 1

  def _collect(self, target, material):
    info = constants.collect.get(material)
    if not info:
      return
    for name, amount in info['require'].items():
      if self.inventory[name] < amount:
        return
    self.world[target] = info['leaves']
    for name, amount in info['receive'].items():
      self.inventory[name] += 1
    self.achievements[f'collect_{material}'] += 1

  def _place(self, name, target, material):
    if not self.is_free(target):
      return
    info = constants.place[name]
    if material not in info['where']:
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.world[target] = name
    self.achievements[f'place_{name}'] += 1

  def _make(self, name):
    nearby = self.world.nearby(self.pos, 2)
    info = constants.make[name]
    if not all(util in nearby for util in info['nearby']):
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.inventory[name] += 1
    self.achievements[f'make_{name}'] += 1


class Cow(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)
    self.health = 3

  @property
  def texture(self):
    return 'cow'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    if self.random.uniform() < 0.5:
      self.move(self.random_dir())


class Zombie(Object):

  def __init__(self, world, pos, player):
    super().__init__(world, pos)
    self.player = player
    self.health = 5
    self.near = False

  @property
  def texture(self):
    return 'zombie'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    dist = self.distance(self.player)
    if dist <= 1:
      if not self.near:
        self.near = True
      elif self.random.uniform() > 0.7:
        self.player.health -= 1
    else:
      self.near = False
    if dist <= 6 and self.random.uniform() < 0.8:
      self.move(self.toward(self.player, self.random.uniform() < 0.7))
    else:
      self.move(self.random_dir())


class Skeleton(Object):

  def __init__(self, world, pos, player):
    super().__init__(world, pos)
    self.player = player
    self.health = 3
    self.reload = 0

  @property
  def texture(self):
    return 'skeleton'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    self.reload = max(0, self.reload - 1)
    dist = self.distance(self.player.pos)
    if dist <= 3:
      moved = self.move(-self.toward(self.player, self.random.uniform() < 0.6))
      if moved:
        return
    if dist <= 5 and self.random.uniform() < 0.5:
      self._shoot(self.toward(self.player))
    elif dist <= 8 and self.random.uniform() < 0.3:
      self.move(self.toward(self.player, self.random.uniform() < 0.6))
    elif self.random.uniform() < 0.2:
      self.move(self.random_dir())

  def _shoot(self, direction):
    if self.reload > 0:
      return
    if direction[0] == 0 and direction[1] == 0:
      return
    pos = self.pos + direction
    if self.is_free(pos, Arrow.walkable):
      self.world.add(Arrow(self.world, pos, direction))
      self.reload = 4


class Arrow(Object):

  def __init__(self, world, pos, facing):
    super().__init__(world, pos)
    self.facing = facing

  @property
  def texture(self):
    return {
        (-1, 0): 'arrow-left',
        (+1, 0): 'arrow-right',
        (0, -1): 'arrow-up',
        (0, +1): 'arrow-down',
    }[tuple(self.facing)]

  @engine.staticproperty
  def walkable():
    return constants.walkable + ['water', 'lava']

  def update(self):
    target = self.pos + self.facing
    material, obj = self.world[target]
    if obj:
      obj.health -= 1
      self.world.remove(self)
    elif material not in self.walkable:
      self.world.remove(self)
    else:
      self.move(self.facing)
