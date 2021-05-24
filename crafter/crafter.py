import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen


class Env:

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64), length=10000, health=5,
      seed=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    unit = size // view
    self._area = area
    self._size = size
    self._length = length
    self._health = health
    self._seed = seed
    self._episode = 0
    self._textures = engine.Textures(constants.root / 'assets')
    self._terrain = engine.Terrain(constants.materials, area)
    self._objs = engine.Objects(area)
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self._terrain, self._objs, self._textures, unit,
        [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, unit, [view[0], item_rows])
    self._border = (size - unit * view) // 2
    self._step = None
    self._player = None
    self._last_health = None
    self._last_achivements = None

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
    self._step = 0
    self._episode += 1
    self._terrain.reset()
    self._objs.reset()
    center = self._area[0] // 2, self._area[1] // 2
    self._player = objects.Player(center, self._health)
    self._objs.add(self._player)
    self._last_health = self._health
    self._last_achivements = self._player.achievements.copy()
    worldgen.generate_world(
        self._terrain, self._objs, center,
        seed=hash((self._seed, self._episode)))
    return self._obs()

  def step(self, action):
    self._step += 1
    for obj in self._objs:
      obj.update(self._terrain, self._objs, self._player, action)
    obs = self._obs()
    reward = 0.0
    if len(self._player.achievements) > len(self._last_achivements):
      self._last_achivements = self._player.achievements.copy()
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
        'health': _uint8(self._player.health),
        'inventory': {k: _uint8(v) for k, v in self._player.inventory.items()},
        'achievements': self._last_achivements.copy(),
        'discount': 1 - float(dead),
    }
    return obs, reward, done, info

  def render(self):
    canvas = np.zeros(tuple(self._size) + (3,), np.uint8)
    local_view = self._local_view(self._player)
    item_view = self._item_view({
        'heart': self._player.health,
        **self._player.inventory})
    view = local_view
    view = np.concatenate([local_view, item_view], 1)
    (x, y), (w, h) = self._border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    return {'image': self.render()}


def _uint8(value):
  return np.array(max(0, min(value, 255)), dtype=np.uint8)
