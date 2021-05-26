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


class staticproperty:

  def __init__(self, function):
    self.function = function

  def __get__(self, instance, owner=None):
    return self.function()


class World:

  def __init__(self, area, seed=None):
    self._terrain = np.zeros(area, np.uint8)
    self._material_names = {0: None}
    self._material_ids = {None: 0}
    self._objects = [None]
    self._coords = np.zeros(area, np.uint32)
    self._random = np.random.RandomState(seed)

  @property
  def area(self):
    return self._terrain.shape

  @property
  def random(self):
    return self._random

  @property
  def objects(self):
    yield from (obj for obj in self._objects if obj)

  def reset(self, seed=None):
    # TODO: Not really needed. Can just create new instance.
    self._random = np.random.RandomState(seed)
    self._terrain[:] = 0
    self._objects = [None]
    self._coords[:] = 0

  def __setitem__(self, pos, material):
    if material not in self._material_ids:
      id_ = len(self._material_ids)
      self._material_ids[material] = id_
      self._material_names[id_] = material
    self._terrain[pos] = self._material_ids[material]

  def __getitem__(self, pos):
    if _inside((0, 0), pos, self.area):
      material = self._material_names[self._terrain[tuple(pos)]]
    else:
      material = None
    if not (0 <= pos[0] < self._coords.shape[0]):
      obj = False
    elif not (0 <= pos[1] < self._coords.shape[1]):
      obj = False
    else:
      obj = self._objects[self._coords[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    # TODO: Return both nearby materials and objects.
    ids = set(self._terrain[
        pos[0] - distance: pos[0] + distance,
        pos[1] - distance: pos[1] + distance].flatten().tolist())
    return tuple(self._material_names[x] for x in ids)

  def count(self, material):
    if material not in self._material_ids:
      return 0
    return (self._terrain == self._material_ids[material]).sum()

  def add(self, obj):
    assert hasattr(obj, 'pos')
    obj.pos = np.array(obj.pos)
    assert self[obj.pos][1] is None
    self._coords[tuple(obj.pos)] = len(self._objects)
    self._objects.append(obj)

  def remove(self, obj):
    self._objects[self._coords[tuple(obj.pos)]] = None
    self._coords[tuple(obj.pos)] = 0

  def move(self, obj, pos):
    pos = np.array(pos)
    assert self[pos][1] is None
    self._coords[tuple(pos)] = self._coords[tuple(obj.pos)]
    self._coords[tuple(obj.pos)] = 0
    obj.pos = pos


class Textures:

  def __init__(self, directory):
    self._originals = {}
    self._textures = {}
    for filename in pathlib.Path(directory).glob('*.png'):
      image = imageio.imread(filename.read_bytes())
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

  def __init__(self, world, textures, unit, grid):
    self._world = world
    self._textures = textures
    self._unit = np.array(unit)
    self._grid = np.array(grid)
    self._offset = self._grid // 2
    self._area = np.array(self._world.area)
    self._center = None

  def __call__(self, player):
    self._center = np.array(player.pos)
    canvas = np.zeros(tuple(self._grid * self._unit) + (3,), np.uint8) + 127
    for x in range(self._grid[0]):
      for y in range(self._grid[1]):
        pos = self._center + np.array([x, y]) - self._offset
        if not _inside((0, 0), pos, self._area):
          continue
        texture = self._textures.get(self._world[pos][0], self._unit)
        _draw(canvas, np.array([x, y]) * self._unit, texture)
    for obj in self._world.objects:
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
    pos = (pos * self._unit + 0.4 * self._unit).astype(np.int32)
    text = str(amount) if amount in (1, 2, 3, 4, 5) else 'unknown'
    texture = self._textures.get(text, 0.6 * self._unit)
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
