import numpy as np

from . import constants
from . import engine


class Object:

  def __init__(self, world, pos):
    self.world = world
    self.pos = np.array(pos)
    self.random = world.random
    self.inventory = {'health': 0}

  @property
  def texture(self):
    raise 'unknown'

  @property
  def walkable(self):
    return constants.walkable

  @property
  def health(self):
    return self.inventory['health']

  @health.setter
  def health(self, value):
    self.inventory['health'] = value

  @property
  def all_dirs(self):
    return ((-1, 0), (+1, 0), (0, -1), (0, +1))

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
    return self.all_dirs[self.random.randint(0, 4)]


class Player(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)
    self.facing = (0, 1)
    self.inventory = {
        name: info['initial'] for name, info in constants.items.items()}
    self.achievements = {name: 0 for name in constants.achievements}
    self._hunger = 0
    self._thirst = 0
    self._fatigue = 0
    self._degen = 0
    self._regen = 0
    self._sleeping = 0

  @property
  def texture(self):
    if self._sleeping:
      return 'player-sleep'
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
    target = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
    material, obj = self.world[target]
    action = constants.actions[action]
    if self._sleeping:
      self._sleeping -= 1
      action = 'noop'
      if self._sleeping == 0:
        self.inventory['energy'] += 1
        if self.inventory['energy'] < constants.items['energy']['max']:
          action = 'sleep'
    if action == 'noop':
      pass
    elif action.startswith('move_'):
      self._move(action[len('move_'):])
    elif action == 'do' and obj:
      self._do_object(obj)
    elif action == 'do':
      self._do_material(target, material)
    elif action == 'sleep':
      self._sleeping = 30
    elif action.startswith('place_'):
      self._place(action[len('place_'):], target, material)
    elif action.startswith('make_'):
      self._make(action[len('make_'):])
    self._update_life_vars()
    for name, amount in self.inventory.items():
      maxmium = constants.items[name]['max']
      self.inventory[name] = max(0, min(amount, maxmium))

  def _update_life_vars(self):
    self._hunger += 1
    if self._hunger >= 50:
      self._hunger = 0
      self.inventory['food'] -= 1

    self._thirst += 1
    if self._thirst >= 50:
      self._thirst = 0
      self.inventory['drink'] -= 1

    self._fatigue += 1
    if self._sleeping:
      self._fatigue = 0
    elif self._fatigue >= 50:
      self._fatigue = 0
      self.inventory['energy'] -= 1

    if self._degen >= 30:
      self._degen = 0
      self.health -= 1

    necessities = (
        self.inventory['food'] > 0,
        self.inventory['drink'] > 0,
        self.inventory['energy'] > 0)
    if all(necessities):
      self._degen = 0
      self._regen += 1
      if self._regen >= 50:
        self.health += 1
        self._regen = 0
    else:
      self._regen = 0
      self._degen += 1

  def _move(self, direction):
    directions = dict(left=(-1, 0), right=(+1, 0), up=(0, -1), down=(0, +1))
    self.facing = directions[direction]
    self.move(self.facing)
    if self.world[self.pos][0] == 'lava':
      self.health = 0

  def _do_object(self, obj):
    damage = max([
        1,
        self.inventory['wood_sword'] and 2,
        self.inventory['stone_sword'] and 3,
        self.inventory['iron_sword'] and 5,
    ])
    if isinstance(obj, Fence):
      self.world.remove(obj)
      self.inventory['fence'] += 1
      self.achievements['collect_fence'] += 1
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
        self.inventory['food'] += 3
        self._hunger = 0
        self.achievements['find_food'] += 1

  def _do_material(self, target, material):
    if material == 'water':
      # TODO: Keep track of previous inventory state to do this in a more
      # general way.
      self._thirst = 0
    info = constants.collect.get(material)
    if not info:
      return
    for name, amount in info['require'].items():
      if self.inventory[name] < amount:
        return
    for name, amount in info['receive'].items():
      self.inventory[name] += amount
    self.world[target] = info['leaves']
    self.achievements[f'collect_{material}'] += 1

  def _place(self, name, target, material):
    if self.world[target][1]:
      return
    info = constants.place[name]
    if material not in info['where']:
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    if info['type'] == 'material':
      self.world[target] = name
    elif info['type'] == 'object':
      cls = {
          'fence': Fence,
      }[name]
      self.world.add(cls(self.world, target))
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
    self.inventory[name] += info['gives']
    self.achievements[f'make_{name}'] += 1


class Cow(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)
    self.health = 3
    # self.fertile = 0

  @property
  def texture(self):
    return 'cow'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    # self.fertile += 1
    if self.random.uniform() < 0.5:
      direction = self.random_dir()
      # self._maybe_mate(direction)
      self.move(direction)

  # def _maybe_mate(self, direction):
  #   if self.fertile < 100:
  #     return
  #   if not isinstance(self.world[self.pos + direction][1], Cow):
  #     return
  #   frees = [self.pos + x for x in self.all_dirs if x != direction]
  #   if any(isinstance(self.world[x][1], Cow) for x in frees):
  #     return
  #   frees = [x for x in frees if self.is_free(x)]
  #   if len(frees) < 2:
  #     return
  #   if self.random.uniform() < 0.1:
  #     self.fertile = 0
  #     target = frees[self.random.randint(0, len(frees))]
  #     self.world.add(Cow(self.world, target))
  #     print('Cows just mated.')


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


class Fence(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)

  @property
  def texture(self):
    return 'fence'

  def update(self):
    pass
