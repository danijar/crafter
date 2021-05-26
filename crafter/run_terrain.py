import argparse

import imageio
import numpy as np

import crafter


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--amount', type=int, default=4)
  parser.add_argument('--cols', type=int, default=4)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--size', type=int, default=1024)
  parser.add_argument('--filename', type=str, default='terrain.png')
  args = parser.parse_args()

  env = crafter.Env(args.area, args.area, args.size, seed=args.seed)
  images = []
  for index in range(args.amount):
    images.append(env.reset())
    diamonds = env._world.count('diamond')
    print(f'Map: {index:>2}, diamonds: {diamonds:>2}')

  rows = len(images) // args.cols
  strips = []
  for row in range(rows):
    strip = []
    for col in range(args.cols):
      try:
        strip.append(images[row * args.cols + col])
      except IndexError:
        strip.append(np.zeros_like(strip[-1]))
    strips.append(np.concatenate(strip, 1))
  grid = np.concatenate(strips, 0)

  imageio.imsave(args.filename, grid)
  print('Saved', args.filename)


if __name__ == '__main__':
  main()
