import argparse

import imageio
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise

import crafter


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--view', type=int, nargs=2, default=(23, 13))
  parser.add_argument('--length', type=int, default=None)
  parser.add_argument('--health', type=int, default=5)
  parser.add_argument('--window', type=int, nargs=2, default=(1280, 720))
  parser.add_argument('--record', type=str, default=None)
  parser.add_argument('--fps', type=int, default=5)
  args = parser.parse_args()

  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_1: 'place_stone',
      pygame.K_2: 'place_table',
      pygame.K_3: 'place_furnace',
      pygame.K_4: 'make_wood_pickaxe',
      pygame.K_5: 'make_stone_pickaxe',
      pygame.K_6: 'make_iron_pickaxe',
  }
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')

  env = crafter.Env(
      args.area, args.view, args.window, args.length, args.health, args.seed)
  env.reset()
  achievements = set()
  return_ = 0
  was_done = False
  if args.record:
    frames = []

  print('Diamonds exist:', env._terrain.count('diamond'))

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
    if len(env._player.achievements) > len(achievements):
      for name in env._player.achievements:
        if name not in achievements:
          count = len(env._player.achievements)
          messages.append(f'Achievement ({count}/13): {name}')
      achievements = env._player.achievements.copy()
    if env._step > 0 and env._step % 100 == 0:
      messages.append(f'Time step: {env._step}')
    if reward:
      messages.append(f'Reward: {reward}')
      return_ += reward
    if done and not was_done:
      was_done = True
      messages.append('Episode done')
    if messages:
      print('\n'.join(messages), sep='')

    if args.record:
      frames.append(obs['image'])
    surface = pygame.surfarray.make_surface(
        obs['image'].transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

  pygame.quit()
  print('Return:', return_)
  if args.record:
    imageio.mimsave(args.record, frames)
    print('Saved', args.record)


if __name__ == '__main__':
  main()
