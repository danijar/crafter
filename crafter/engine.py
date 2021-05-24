import collections
import pathlib

import imageio
import numpy as np
from PIL import Image


DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
DictSpace = collections.namedtuple('DictSpace', 'spaces')


class Terrain:

  def __init__(self, materials, area):
    self._map = np.zeros(area, np.uint8)
    self._names = {i + 1: n for i, n in enumerate(materials)}
    self._ids = {n: i + 1 for i, n in enumerate(materials)}

  @property
  def area(self):
    return self._map.shape

  def reset(self):
    self._map[:] = 0

  def count(self, material):
    return (self._map == self._ids[material]).sum()

  def nearby(self, pos, distance):
    ids = set(self._map[
        pos[0] - distance: pos[0] + distance,
        pos[1] - distance: pos[1] + distance].flatten().tolist())
    return tuple(self._names[x] for x in ids)

  def __setitem__(self, pos, material):
    self._map[pos] = self._ids[material]

  def __getitem__(self, pos):
    return self._names[self._map[tuple(pos)]]


class Objects:

  def __init__(self, area):
    self._objects = [None]
    self._map = np.zeros(area, np.uint32)

  def reset(self):
    self._objects = [None]
    self._map[:] = 0

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

  def nearby(self, pos, types=None):
    raise NotImplementedError

  def at(self, pos):
    if not (0 <= pos[0] < self._map.shape[0]):
      return False
    if not (0 <= pos[1] < self._map.shape[1]):
      return False
    return self._objects[self._map[pos]]


class Textures:

  def __init__(self, filenames, size):
    self._textures = {}
    for name, filename in filenames.items():
      filename = pathlib.Path(__file__).parent / filename
      image = imageio.imread(filename)
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      image = np.array(Image.fromarray(image).resize(
          size, resample=Image.NEAREST))
      self._textures[name] = image

  def __getitem__(self, name):
    return self._textures[name]


class Actions:

  pass


class Recipes:

  pass


class GlobalView:

  pass


class LocalView:

  def __init__(self, terrain, objects, textures, grid, view):
    self._terrain = terrain
    self._objects = objects
    self._textures = textures
    self._grid = np.array(grid)
    self._view = np.array(view)
    self._shape = self._grid * (2 * self._view + 1)
    self._area = np.array(self._terrain.area)
    self._center = None

  def __call__(self, player):
    self._center = np.array(player.pos)
    canvas = np.zeros(tuple(self._shape) + (3,), np.uint8) + 127
    for x in range(2 * self._view[0] + 1):
      for y in range(2 * self._view[1] + 1):
        pos = self._center + np.array([x, y]) - self._view
        if not _inside((0, 0), pos, self._area):
          continue
        texture = self._textures[self._terrain[pos]]
        self._draw(canvas, np.array([x, y]) * self._grid, texture)
    for obj in self._objects:
      pos = obj.pos - self._center + self._view
      if not _inside((0, 0), pos, 2 * self._view + 1):
        continue
      texture = self._textures[obj.texture]
      self._draw_alpha(canvas, pos * self._grid, texture)
    return canvas.transpose((1, 0, 2))

  def _draw(self, canvas, pos, texture):
    (x, y), (w, h) = pos, self._grid
    if texture.shape[-1] == 4:
      texture = texture[..., :3]
    canvas[x: x + w, y: y + h] = texture

  def _draw_alpha(self, canvas, pos, texture):
    (x, y), (w, h) = pos, self._grid
    if texture.shape[-1] == 4:
      alpha = texture[..., 3:].astype(np.float32) / 255
      texture = texture[..., :3].astype(np.float32) / 255
      current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
      blended = alpha * texture + (1 - alpha) * current
      texture = (255 * blended).astype(np.uint8)
    canvas[x: x + w, y: y + h] = texture


class UncoverView:

  pass


class InventoryView:

  pass


def _inside(lhs, mid, rhs):
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])
