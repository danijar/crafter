# Crafter

[![PyPI](https://img.shields.io/pypi/v/crafter.svg)](https://pypi.python.org/pypi/crafter/#history)

Open world survival environment for reinforcement learning.

![Crafter Terrain](https://github.com/danijar/crafter/raw/main/media/terrain.png)

## Highlights

Crafter is a procedurally generated 2D world, where the agent finds food, and
collect materials to build tools, which in turn unlock new materials, all while
avoiding or defending against monsters.

- **Generalization:** New procedurally generated map for each episode.
- **Exploration:** Materials unlock new tools which unlock new materials.
- **Memory:** Input images show small part of the world centered at the agent.
- **Nontrivial survival:** Must find food and avoid or defend against zombies.
- **Easy usage:** Image inputs and flat categorical action space.
- **Fast:** Runs at 2000 FPS on a simple laptop.
- **Reproducible:** All randomness is controlled by a seed.

## Play Yourself

You can play the game yourself with an interactive window and keyboard input.
The mapping from keys to actions, health level, and inventory state are printed
to the terminal.

```sh
# Install with GUI
pip3 install 'crafter[gui]'

# Start the game
crafter

# Alternative way to start the game
python3 -m crafter.run_gui
```

![Crafter Video](https://github.com/danijar/crafter/raw/main/media/video.gif)

The following optional command line flags are available:

| Flag | Default | Description |
| :--- | :-----: | :---------- |
| `--window <width> <height>` | 1280 720 | Window size in pixels, used as width and height. |
| `--fps <integer>` | 5 | How many times to update the environment per second. |
| `--record <filename>.mp4` | None | Record a video of the trajectory. |
| `--area <width> <height>` | 64 64 | The size of the world in cells. |
| `--view <width> <height>` | 23 13 | The layout size in cells; determines view distance. |
| `--length <integer>` | None | Time limit for the episode. |
| `--seed <integer>` | None | Determines world generation and creatures. |

## Training Agents

Installation: `pip3 install -U crafter`

The environment follows the [OpenAI Gym][gym] interface:

```py
import crafter

env = crafter.Env(seed=0)
obs = env.reset()
assert obs.shape == (64, 64, 3)

done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

[gym]: https://github.com/openai/gym

## Environment Details

### Constructor

For comparability between papers, we recommend using the environment in its
default configuration. Nonetheless, the environment can be configured via its
constructor:

```py
crafter.Env(area=(64, 64), view=(9, 9), size=(64, 64), length=10000, seed=None)
```

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `area` | `(64, 64)` | Size of the world in grid cells. |
| `view` | `(9, 9)` | Layout size in cells; determines view distance. |
| `size` | `(64, 64)` | Render size of the images in pixels. |
| `length` | `10000` | Time limit for the episode, can be `None`. |
| `health` | `5` | Initial health level of the player. |
| `seed` | None | Interger that determines world generation and creatures. |

### Reward

The reward can either be given to the agent or used as a proxy metric for
evaluating unsupervised agents.

The reward is +1 when the agent unlocks a new achievement, -0.1 when its health
level decreases, +0.1 when it increases, and 0 for all other time steps. The
achievements are as follows:

- `collect_coal`
- `collect_diamond`
- `collect_iron`
- `collect_stone`
- `collect_tree`
- `defeat_zombie`
- `defeat_skeleton`
- `find_food`
- `make_iron_pickaxe`
- `make_stone_pickaxe`
- `make_wood_pickaxe`
- `make_iron_sword`
- `make_stone_sword`
- `make_wood_sword`
- `place_furnace`
- `place_stone`
- `place_table`

The sum of rewards per episode can range from -0.5 (losing all health without
any achivements) to 17 (unlocking all achievements and keeping or restoring all
health).

### Termination

The episode terminates when the health points of the agent reach zero. Episodes
also end when reaching a time limit, which is 10000 steps by default.

### Observation Space

Each observation is an RGB image that shows a local view of the world around
the player, as well as the health counter and inventory state of the agent.

### Action Space

The action space is categorical. Each action is an integer index representing
one of the possible actions:

| Integer | Name | Requirement |
| :-----: | :--- | :---------- |
| 0 | `noop` | Always applicable. |
| 1 | `move_left` | Flat ground left to the agent. |
| 2 | `move_right` | Flat ground right to the agent. |
| 3 | `move_up` | Flat ground above the agent. |
| 4 | `move_down` | Flat ground below the agent. |
| 5 | `do` | Facing creature or material and have necessary tool. |
| 6 | `place_stone` | Stone in inventory. |
| 7 | `place_table` | Wood in inventory. |
| 8 | `place_furnace` | Stone in inventory. |
| 9 | `make_wood_pickaxe` | Nearby table. Wood in inventory. |
| 10 | `make_stone_pickaxe` | Nearby table. Wood, stone in inventory. |
| 11 | `make_iron_pickaxe` | Nearby table, furnace. Wood, coal, iron an inventory. |
| 12 | `make_wood_sword` | Nearby table. Wood in inventory. |
| 13 | `make_stone_sword` | Nearby table. Wood, stone in inventory. |
| 14 | `make_iron_sword` | Nearby table, furnace. Wood, coal, iron an inventory. |

### Info Dictionary

The step function returns an `info` directionary with additional information
about the environment state. It can be used for evaluation and debugging but
should not be provided to the agent. The following entries are available:

| Key | Type | Description |
| :-- | :--: | :---------- |
| `health` | uint8 | Health counter of the player. |
| `inventory` | dict | Mapping from item names to inventory counts. |
| `achievements` | dict | Mapping from achievement names to their counts. |
| `discount` | float | 1 during the episode and 0 at the last step. |

## Baselines

To understand how challenging the environment is, we trained the
[DreamerV2][dreamerv2] agent 10 times for 30M environment steps each. The agent
receives the rewards that correspond to the 13 achievements that can be
unlocked in each episode, the most difficult of which is to collect a diamond.

![Crafter Terrain](https://github.com/danijar/crafter/raw/main/media/dreamerv2.png)

We observe consistent learning progress. Eventually, many of the runs
sporadically collect a diamond. This shows that the environment is challenging
and unsolved but not completely out of reach.

[dreamerv2]: https://github.com/danijar/dreamerv2

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/crafter/issues
