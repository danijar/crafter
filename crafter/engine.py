import collections
import pathlib

import imageio
import numpy as np
from PIL import Image


DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
DictSpace = collections.namedtuple('DictSpace', 'spaces')


class AttrDict(dict):

  __getattr__ = dict.__getitem__


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
    if not _inside((0, 0), pos, self.area):
      return None
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

  def __init__(self, directory):
    self._originals = {}
    self._textures = {}
    for filename in pathlib.Path(directory).glob('*.png'):
      image = imageio.imread(filename)
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      self._originals[filename.stem] = image
      self._textures[(filename.stem, image.shape[:2])] = image

  def get(self, name, size):
    size = int(size[0]), int(size[1])
    key = name, size
    if key not in self._textures:
      image = self._originals[name]
      image = Image.fromarray(image)
      image = image.resize(size[::-1], resample=Image.NEAREST)
      image = np.array(image)
      self._textures[key] = image
    return self._textures[key]


class GlobalView:

  pass


class UncoverView:

  pass


class LocalView:

  def __init__(self, terrain, objects, textures, unit, grid):
    self._terrain = terrain
    self._objects = objects
    self._textures = textures
    self._unit = np.array(unit)
    self._grid = np.array(grid)
    self._offset = self._grid // 2
    self._area = np.array(self._terrain.area)
    self._center = None

  def __call__(self, player):
    self._center = np.array(player.pos)
    canvas = np.zeros(tuple(self._grid * self._unit) + (3,), np.uint8) + 127
    for x in range(self._grid[0]):
      for y in range(self._grid[1]):
        pos = self._center + np.array([x, y]) - self._offset
        if not _inside((0, 0), pos, self._area):
          continue
        texture = self._textures.get(self._terrain[pos], self._unit)
        _draw(canvas, np.array([x, y]) * self._unit, texture)
    for obj in self._objects:
      pos = obj.pos - self._center + self._offset
      if not _inside((0, 0), pos, self._grid):
        continue
      texture = self._textures.get(obj.texture, self._unit)
      _draw_alpha(canvas, pos * self._unit, texture)
    return canvas


class ItemView:

  def __init__(self, textures, unit, grid):
    self._textures = textures
    self._unit = np.array(unit)
    self._grid = np.array(grid)

  def __call__(self, inventory):
    canvas = np.zeros(tuple(self._grid * self._unit) + (3,), np.uint8)
    for index, (item, amount) in enumerate(inventory.items()):
      if amount < 1:
        continue
      self._item(canvas, index, item)
      self._amount(canvas, index, amount)
    return canvas

  def _item(self, canvas, index, item):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * self._unit + 0.1 * self._unit).astype(np.int32)
    texture = self._textures.get(item, 0.8 * self._unit)
    _draw_alpha(canvas, pos, texture)

  def _amount(self, canvas, index, amount):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * self._unit + 0.5 * self._unit).astype(np.int32)
    text = str(amount) if amount in (1, 2, 3, 4, 5) else 'unknown'
    texture = self._textures.get(text, 0.5 * self._unit)
    _draw_alpha(canvas, pos, texture)


def _inside(lhs, mid, rhs):
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])

def _draw(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    texture = texture[..., :3]
  canvas[x: x + w, y: y + h] = texture

def _draw_alpha(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    alpha = texture[..., 3:].astype(np.float32) / 255
    texture = texture[..., :3].astype(np.float32) / 255
    current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
    blended = alpha * texture + (1 - alpha) * current
    texture = (255 * blended).astype(np.uint8)
  canvas[x: x + w, y: y + h] = texture
