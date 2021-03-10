# Crafter

Open world survival environment for reinforcement learning.

## Highlights

Crafter is a procedurally generated 2D world, where the agent finds food,
avoids or defends against zombies, and collect materials to build tools, which
in turn unlock new materials.

- **Generalization:** New procedurally generated map for each episode.
- **Exploration:** Materials unlock new tools which unlock new materials.
- **Memory:** Input images show small part of the world centered at the agent.
- **No trivial behaviors:** Must find food and avoid or defend against zombies.
- **Easy:** Flat categorical action space with 12 actions.
- **Fast:** Runs at 2000 FPS on a simple laptop.
- **Reproducible:** All randomness is controlled by a seed.

## Instructions

Installation: `pip3 install -U crafter`

The environment follows the [OpenAI Gym][gym] interface:

```py
import crafter

env = crafter.Env(seed=0)
obs = env.reset()
assert obs['image'].shape == (64, 64, 3)

done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

[gym]: https://github.com/openai/gym

## Play Yourself

You can play the game yourself with an interactive window and keyboard input.
The mapping from keys to actions, health level, and inventory state are printed
to the terminal.

```sh
# Install the GUI
pip3 install 'crafter[gui]'

# Start the game
crafter

# Alternative way to start the game
python3 -m crafter.run_gui
```

The following optional command line flags are available:

| Flag | Default | Description |
| :--- | :-----: | :---------- |
| `--window <size>` | 500 | Window size in pixels, used as width and height. |
| `--fps <integer>` | 3 | How many times to update the environment per second. |
| `--record <filename>.mp4` | None | Record a video of the trajectory. |
| `--map <width> <height>` | 64 64 | The size of the world in cells. |
| `--view <distance>` | 4 | The view distance of the player in cells. |
| `--length <integer>` | None | Time limit for the episode. |
| `--seed <integer>` | None | Determines world generation and creatures. |

## Environment Details

### Constructor

For comparability between papers, we recommend using the environment in its
default configuration. Nonetheless, the environment can be configured via its
constructor:

```py
crafter.Env(area=(64, 64), view=5, size=64, length=100000, seed=None)
```

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `area` | `(64, 64)` | Size of the world in cells. |
| `view` | `5` | View distance of the player in cells. |
| `size` | `64` | Render size of the images, used for both width and height. |
| `length` | `100000` | Time limit for the episode, can be `None`. |
| `seed` | None | Interger that determines world generation and creatures. |

### Reward

The reward is sparse. It is 1 when the agent collects a new material or creates
a new object for the first time and 0 for all other steps. The reward can
either be given to the agent or used as a proxy metric for evaluating
unsupervised agents.

### Termination

The episode terminates when the health points of the agent reach zero. Episodes
also end when reaching a time limit, which is 100,000 steps by default.

### Observation Space

Each observation is a dictionary that contains a local image centered at the
agent and counters for player health and inventory. The following keys are
available:

| Key | Space |
| :-- | :---- |
| `image` | `Box(0, 255, (64, 64, 3), np.uint8)` |
| `health` | `Box(0, 255, (), np.uint8)` |
| `wood` | `Box(0, 255, (), np.uint8)` |
| `stone` | `Box(0, 255, (), np.uint8)` |
| `diamond` | `Box(0, 255, (), np.uint8)` |
| `wood_pickaxe` | `Box(0, 255, (), np.uint8)` |
| `stone_pickaxe` | `Box(0, 255, (), np.uint8)` |
| `iron_pickaxe` | `Box(0, 255, (), np.uint8)` |

### Action Space

The action space is categorical. Each action is an integer index representing
one of the 12 possible actions:

| Integer | Name | Requirement |
| :-----: | :--- | :---------- |
| 0 | `noop` | Always applicable. |
| 1 | `left` | Flat ground left to the agent. |
| 2 | `right` | Flat ground right to the agent. |
| 3 | `up` | Flat ground above the agent. |
| 4 | `down` | Flat ground below the agent. |
| 5 | `grab_or_attack` | Facing creature or material and has necessary tool. |
| 6 | `place_stone` | Stone in inventory. |
| 7 | `place_table` | Wood in inventory. |
| 8 | `place_furnace` | Stone in inventory. |
| 9 | `make_wood_pickaxe` | Nearby table and wood in inventory. |
| 10 | `make_stone_pickaxe` | Nearby tabel and wood, stone in inventory. |
| 11 | `make_iron_pickaxe` | Nearby furnace and wood, coal, iron an inventory. |

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/crafter/issues
