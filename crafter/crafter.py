import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen


class Env:

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64), length=10000,
      seed=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    unit = size // view
    self._area = area
    self._size = size
    self._length = length
    self._seed = seed
    self._episode = 0

    self._world = engine.World(area, constants.materials)

    self._textures = engine.Textures(constants.root / 'assets')

    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self._world, self._textures, unit,
        [view[0], view[1] - item_rows])

    self._item_view = engine.ItemView(
        self._textures, unit, [view[0], item_rows])
    self._border = (size - unit * view) // 2

    self._step = None
    self._player = None
    self._last_health = None
    self._unlocked = None

  @property
  def observation_space(self):
    return engine.BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return engine.DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self._world.reset(seed=hash((self._seed, self._episode)) % 2 ** 32)
    self._player = objects.Player(self._world, center)
    self._last_health = self._player.health
    self._world.add(self._player)
    self._unlocked = set()
    worldgen.generate_world(self._world, self._player)
    return self._obs()

  def step(self, action):
    self._step += 1
    self._player.action = constants.actions[action]
    for obj in self._world.objects:
      obj.update()
    # print('')
    # TODO: This should be run less frequently than every step.
    for chunk, objs in self._world.chunks.items():
      self._balance_chunk(chunk, objs)
    obs = self._obs()
    reward = 0.0
    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    if unlocked:
      self._unlocked |= unlocked
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
        'health': self._player.health,
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'discount': 1 - float(dead),
    }
    return obs, reward, done, info

  def render(self):
    canvas = np.zeros(tuple(self._size) + (3,), np.uint8)
    local_view = self._local_view(self._player)
    item_view = self._item_view(self._player.inventory)
    view = local_view
    view = np.concatenate([local_view, item_view], 1)
    (x, y), (w, h) = self._border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    return self.render()

  def _balance_chunk(self, chunk, objs):
    # xmin, xmax, ymin, ymax = chunk
    # zombies = sum(1 for obj in objs if isinstance(obj, objects.Zombie))
    # print(chunk, 'zombies', zombies)
    pass
