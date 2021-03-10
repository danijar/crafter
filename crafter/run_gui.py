import argparse

import imageio
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise

import crafter


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--map', nargs=2, type=int, default=(64, 64))
parser.add_argument('--view', type=int, default=4)
parser.add_argument('--length', type=int, default=None)
parser.add_argument('--window', type=int, default=500)
parser.add_argument('--record', type=str, default=None)
parser.add_argument('--fps', type=int, default=3)
args = parser.parse_args()

keymap = {
    pygame.K_a: 'left',
    pygame.K_d: 'right',
    pygame.K_w: 'up',
    pygame.K_s: 'down',
    pygame.K_SPACE: 'grab_or_attack',
    pygame.K_1: 'place_stone',
    pygame.K_2: 'place_table',
    pygame.K_3: 'place_furnace',
    pygame.K_4: 'make_wood_pickaxe',
    pygame.K_5: 'make_stone_pickaxe',
    pygame.K_6: 'make_iron_pickaxe',
}
print('\nActions:')
for key, action in keymap.items():
  print(f'  {pygame.key.name(key)}: {action}')

env = crafter.Env(args.map, args.view, args.window, args.length, args.seed)
env.reset()
inventory = None
health = None
if args.record:
  frames = []

pygame.init()
screen = pygame.display.set_mode([args.window, args.window])
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

  obs, _, _, _ = env.step(env.action_names.index(action))
  if not health or health != env._player.health:
    health = env._player.health
    print('\nHealth:', health)
  if not inventory or inventory != env._player.inventory:
    inventory = env._player.inventory.copy()
    print('\nInventory:')
    for key, value in inventory.items():
      print(f'  {key}: {value}')

  if args.record:
    frames.append(obs['image'].transpose((1, 0, 2)))
  surface = pygame.surfarray.make_surface(obs['image'])
  screen.blit(surface, (0, 0))
  pygame.display.flip()
  clock.tick(args.fps)

pygame.quit()
if args.record:
  imageio.mimsave(args.record, frames)
  print('Saved', args.record)
