import argparse

import numpy as np
import imageio
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise
from PIL import Image

import crafter


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
  parser.add_argument('--length', type=int, default=None)
  parser.add_argument('--health', type=int, default=9)
  parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
  parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
  parser.add_argument('--record', type=str, default=None)
  parser.add_argument('--fps', type=int, default=5)
  parser.add_argument('--death', type=str, default='reset', choices=[
      'continue', 'reset', 'quit'])
  args = parser.parse_args()

  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_TAB: 'sleep',

      pygame.K_r: 'place_stone',
      pygame.K_t: 'place_table',
      pygame.K_f: 'place_furnace',
      pygame.K_p: 'place_plant',

      pygame.K_1: 'make_wood_pickaxe',
      pygame.K_2: 'make_stone_pickaxe',
      pygame.K_3: 'make_iron_pickaxe',
      pygame.K_4: 'make_wood_sword',
      pygame.K_5: 'make_stone_sword',
      pygame.K_6: 'make_iron_sword',
  }
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')

  crafter.constants.items['health']['max'] = args.health
  crafter.constants.items['health']['initial'] = args.health

  size = list(args.size)
  size[0] = size[0] or args.window[0]
  size[1] = size[1] or args.window[1]

  env = crafter.Env(args.area, args.view, size, args.length, args.seed)
  env.reset()
  achievements = set()
  duration = 0
  return_ = 0
  was_done = False
  if args.record:
    frames = []
  print('Diamonds exist:', env._world.count('diamond'))

  pygame.init()
  screen = pygame.display.set_mode(args.window)
  clock = pygame.time.Clock()
  running = True
  while running:

    action = None
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        action = 'noop'

    messages = []
    obs, reward, done, _ = env.step(env.action_names.index(action))

    unlocked = {
        name for name, count in env._player.achievements.items()
        if count > 0 and name not in achievements}
    for name in unlocked:
      achievements |= unlocked
      total = len(env._player.achievements.keys())
      messages.append(f'Achievement ({len(achievements)}/{total}): {name}')
    if env._step > 0 and env._step % 100 == 0:
      messages.append(f'Time step: {env._step}')
    if reward:
      messages.append(f'Reward: {reward}')
      return_ += reward
    if done and not was_done:
      was_done = True
      messages.append('Episode done!')
      if args.death == 'quit':
        running = False
      if args.death == 'reset':
        env.reset()
        achievements = set()
        was_done = False
        duration = 0
        return_ = 0
      if args.death == 'continue':
        pass
    if messages:
      print('\n'.join(messages), sep='')

    duration += 1
    if args.record:
      frames.append(obs)
    if size != args.window:
      obs = Image.fromarray(obs)
      obs = obs.resize(args.window, resample=Image.NEAREST)
      obs = np.array(obs)
    surface = pygame.surfarray.make_surface(obs.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

  pygame.quit()
  print('Duration:', duration)
  print('Return:', return_)
  if args.record:
    imageio.mimsave(args.record, frames)
    print('Saved', args.record)


if __name__ == '__main__':
  main()
