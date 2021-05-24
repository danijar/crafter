import functools

import numpy as np
import opensimplex

from . import constants
from . import objects


def generate_world(terrain, objs, center, seed):
  random = np.random.RandomState(seed=np.uint32(seed))
  simplex = opensimplex.OpenSimplex(seed=seed)
  for x in range(terrain.area[0]):
    for y in range(terrain.area[1]):
      _set_terrain(terrain, (x, y), center, random, simplex)
  for x in range(terrain.area[0]):
    for y in range(terrain.area[1]):
      _set_object(terrain, objs, (x, y), center, random)


def _set_terrain(terrain, pos, center, random, simplex):
  x, y = pos
  simplex = functools.partial(_simplex, simplex)
  uniform = random.uniform
  start = 4 - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
  start += 2 * simplex(x, y, 8, 3)
  start = 1 / (1 + np.exp(-start))
  water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
  water -= 2 * start
  mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
  mountain -= 4 * start + 0.3 * water
  if start > 0.5:
    terrain[x, y] = 'grass'
  elif mountain > 0.15:
    if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
      terrain[x, y] = 'path'
    elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
      terrain[x, y] = 'path'
    elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
      terrain[x, y] = 'path'
    elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
      terrain[x, y] = 'coal'
    elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
      terrain[x, y] = 'iron'
    elif mountain > 0.18 and uniform() > 0.995:
      terrain[x, y] = 'diamond'
    elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.4:
      terrain[x, y] = 'lava'
    else:
      terrain[x, y] = 'stone'
  elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
    terrain[x, y] = 'sand'
  elif 0.3 < water:
    terrain[x, y] = 'water'
  else:  # grassland
    if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
      terrain[x, y] = 'tree'
    else:
      terrain[x, y] = 'grass'


def _set_object(terrain, objs, pos, center, random):
  x, y = pos
  uniform = random.uniform
  dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
  if terrain[x, y] in constants.walkable:
    grass = (terrain[x, y] == 'grass')
    if dist > 3 and grass and uniform() > 0.98:
      objs.add(objects.Cow((x, y), random))
    elif dist > 6 and uniform() > 0.993:
      objs.add(objects.Zombie((x, y), random))


def _simplex(simplex, x, y, z, sizes, normalize=True):
  if not isinstance(sizes, dict):
    sizes = {sizes: 1}
  value = 0
  for size, weight in sizes.items():
    value += weight * simplex.noise3d(x / size, y / size, z)
  if normalize:
    value /= sum(sizes.values())
  return value
