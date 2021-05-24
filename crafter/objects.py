import numpy as np

from . import constants


class Player:

  def __init__(self, pos, health):
    self.pos = pos
    self.face = (0, 1)
    self.health = health
    self.inventory = {item: 0 for item in constants.items}
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

  def update(self, terrain, objs, player, action):
    self._hunger += 1
    if self._hunger > 100:
      self.health -= 1
      self._hunger = 0
    target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
    material = terrain[target] or 'end_of_world'
    obj = objs.at(target)
    action = constants.actions[action]
    if action == 'noop':
      pass
    elif action.startswith('move_'):
      self._move(action[len('move_'):], terrain, objs)
    elif action == 'do' and obj:
      self._interact(obj)
    elif action == 'do':
      self._collect(terrain, target, material)
    elif action.startswith('place_'):
      self._place(action[len('place_'):], terrain, target, material)
    elif action.startswith('make_'):
      self._make(action[len('make_'):], terrain.nearby(self.pos, 2))
    for item, amount in self.inventory.items():
      self.inventory[item] = max(0, min(amount, 5))

  def _move(self, direction, terrain, objs):
    directions = dict(left=(-1, 0), right=(+1, 0), up=(0, -1), down=(0, +1))
    self.face = directions[direction]
    target = (self.pos[0] + self.face[0], self.pos[1] + self.face[1])
    if _is_free(target, terrain, objs, constants.walkable):
      objs.move(self, target)
    elif _is_free(target, terrain, objs, ['lava']):
      objs.move(self, target)
      self.health = 0

  def _interact(self, obj):
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

  def _collect(self, terrain, target, material):
    info = constants.collect.get(material)
    if not info:
      return
    for name, amount in info['require'].items():
      if self.inventory[name] < amount:
        return
    terrain[target] = info['leaves']
    for name, amount in info['receive'].items():
      self.inventory[name] += 1
    self.achievements.add(f'collect_{material}')

  def _place(self, name, terrain, target, material):
    info = constants.place[name]
    if material not in info['where']:
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    terrain[target] = name
    self.achievements.add(f'place_{name}')

  def _make(self, name, nearby):
    info = constants.make[name]
    if not all(util in nearby for util in info['nearby']):
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.inventory[name] += 1
    self.achievements.add(f'make_{name}')


class Cow:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random

  @property
  def texture(self):
    return 'cow'

  def update(self, terrain, objs, player, action):
    if self.health <= 0:
      objs.remove(self)
    if self._random.uniform() < 0.5:
      return
    direction = _random_direction(self._random)
    x = self.pos[0] + direction[0]
    y = self.pos[1] + direction[1]
    if _is_free((x, y), terrain, objs, constants.walkable):
      objs.move(self, (x, y))


class Zombie:

  def __init__(self, pos, random):
    self.pos = pos
    self.health = 1
    self._random = random
    self._near = False

  @property
  def texture(self):
    return 'zombie'

  def update(self, terrain, objs, player, action):
    if self.health <= 0:
      objs.remove(self)
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
    if _is_free((x, y), terrain, objs, constants.walkable):
      objs.move(self, (x, y))


def _is_free(pos, terrain, objs, valid):
  if not (0 <= pos[0] < terrain.area[0]):
    return False
  if not (0 <= pos[1] < terrain.area[1]):
    return False
  if terrain[pos] not in valid:
    return False
  if not objs.free(pos):
    return False
  return True


def _random_direction(random):
  if random.uniform() > 0.5:
    return (0, random.randint(-1, 2))
  else:
    return (random.randint(-1, 2), 0)
