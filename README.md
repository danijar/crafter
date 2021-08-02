# Crafter

[![PyPI](https://img.shields.io/pypi/v/crafter.svg)](https://pypi.python.org/pypi/crafter/#history)

Open world survival environment for reinforcement learning.

![Crafter Terrain](https://github.com/danijar/crafter/raw/main/media/terrain.png)

If you find this code useful, please reference in your paper:

```
@misc{hafner2021crafter,
  title = {Crafter: An Open World Survival Benchmark},
  author = {Danijar Hafner},
  year = {2021},
  howpublished = {\url{https://github.com/danijar/crafter}},
}
```

## Highlights

Crafter is a simulated environment that tests a variety of general abilities of
learning agents. It features a randomized open-ended world with image inputs
where the player discovers resources and tools, all while ensuring its own
survival.

- **Generalization:** New procedurally generated map for each episode.
- **Exploration:** Materials unlock new tools which in turn unlock new materials.
- **Long dependencies:** Gathering resources, building shelter, growing fruits.
- **Partial observability:** Each input image reveals only a small part of the world.
- **Survival:** Must find food and water, shelter to rest, defend against monsters.
- **Easy to use:** Pure Python, windowless rendering, few dependencies, flat action space.

## Status

Crafter has been released in its stable version v1.x.x and can now be used for
research projects. The environment will not change anymore to ensure
comparability. Future changes may add additional debugging features that do not
affect the environment itself.

## Play Yourself

You can play the game yourself with an interactive window and keyboard input.
The mapping from keys to actions, health level, and inventory state are printed
to the terminal.

```sh
# Install with GUI
pip3 install pygame
pip3 install crafter

# Start the game
crafter

# Alternative way to start the game
python3 -m crafter.run_gui
```

![Crafter Video](https://github.com/danijar/crafter/raw/main/media/video.gif)

The following optional command line flags are available:

| Flag | Default | Description |
| :--- | :-----: | :---------- |
| `--record <directory>` | None | Directory for recording trajectories in NPZ and MP4 formats. |
| `--window <width> <height>` | 600 600 | Window size in pixels. |
| `--area <width> <height>` | 64 64 | The number of grid cells of the generated world. |
| `--view <width> <height>` | 9 9 | The number of grid cells that are visible in the images. |
| `--size <width> <height>` | 0 0 | Render resolution; defaults to the window size. Setting this to `64 64` shows the low resolution graphics that artificial agents see. |
| `--wait <boolean>` | False | Pauses the game while the player chooses their action. |
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

To ensure comparability across research papers, we recommend using the
environment in its default configuration. Nonetheless, the environment can be
configured via its constructor:

```py
crafter.Env(area=(64, 64), view=(9, 9), size=(64, 64), length=10000, seed=None)
```

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `area` | `(64, 64)` | Size of the world in grid cells. |
| `view` | `(9, 9)` | Layout size in cells; determines view distance. |
| `size` | `(64, 64)` | Render size of the images in pixels. |
| `length` | `10000` | Time limit for the episode, can be `None`. |
| `seed` | None | Interger that determines world generation and creatures. |

### Reward

The reward can either be given to the agent or used as a proxy metric for
evaluating unsupervised agents.

The reward is +1 when the agent unlocks a new achievement, -0.1 when its health
level decreases, +0.1 when it increases, and 0 for all other time steps. The
achievements are as follows:

- `collect_coal`
- `collect_diamond`
- `collect_drink`
- `collect_iron`
- `collect_sapling`
- `collect_stone`
- `collect_wood`
- `defeat_skeleton`
- `defeat_zombie`
- `eat_cow`
- `eat_plant`
- `make_iron_pickaxe`
- `make_iron_sword`
- `make_stone_pickaxe`
- `make_stone_sword`
- `make_wood_pickaxe`
- `make_wood_sword`
- `place_furnace`
- `place_plant`
- `place_stone`
- `place_table`
- `wake_up`

The sum of rewards per episode can range from -0.9 (losing all health without
any achievements) to 22 (unlocking all achievements and keeping or restoring
all health until the time limit is reached). A score of 21.1 or higher means
that all achievements have been unlocked.

### Termination

The episode terminates when the health points of the agent reach zero. Episodes
also end when reaching a time limit, which is 10000 steps by default.

### Observation Space

Each observation is an RGB image that shows a local view of the world around
the player, as well as the life statistics and inventory state of the agent.

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
| 6 | `sleep` | Energy level is below maximum. |
| 7 | `place_stone` | Stone in inventory. |
| 8 | `place_table` | Wood in inventory. |
| 9 | `place_furnace` | Stone in inventory. |
| 10 | `place_plant` | Sapling in inventory. |
| 11 | `make_wood_pickaxe` | Nearby table. Wood in inventory. |
| 12 | `make_stone_pickaxe` | Nearby table. Wood, stone in inventory. |
| 13 | `make_iron_pickaxe` | Nearby table, furnace. Wood, coal, iron an inventory. |
| 14 | `make_wood_sword` | Nearby table. Wood in inventory. |
| 15 | `make_stone_sword` | Nearby table. Wood, stone in inventory. |
| 16 | `make_iron_sword` | Nearby table, furnace. Wood, coal, iron an inventory. |

### Info Dictionary

The step function returns an `info` directionary with additional information
about the environment state. It can be used for evaluation and debugging but
should not be provided to the agent. The following entries are available:

| Key | Type | Description |
| :-- | :--: | :---------- |
| `inventory` | dict | Mapping from item names to inventory counts. |
| `achievements` | dict | Mapping from achievement names to their counts. |
| `discount` | float | 1 during the episode and 0 at the last step. |
| `semantic` | np.array | Categorical representation of the world. |
| `player_pos` | tuple | X and Y position of the player in the world. |

## Baselines

Crafter is designed to be challenging for current learning algorithms but not
completely out of reach. To verify how challenging the environment is, we
trained the [DreamerV2][dreamerv2] agent on Crafter with rewards. We recommend
training for 5M environment steps and reporting the mean score. During this
time, the agent makes consistent learning progress.

<img src="https://github.com/danijar/crafter/raw/main/media/baseline.png" width="500"/>

When training over 10 times longer, the agent also rarely unlocks all
achievements during an episode, including finding a diamond. The open research
challenge ahead of us is to drastically accelerate the exploration and learning
progress and increase the average score by consistently unlocking all
achievements.

[dreamerv2]: https://github.com/danijar/dreamerv2

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/crafter/issues
