**Status:** Stable release

[![PyPI](https://img.shields.io/pypi/v/crafter.svg)](https://pypi.python.org/pypi/crafter/#history)

# Crafter

Open world survival environment for reinforcement learning.

![Crafter Terrain](https://github.com/danijar/crafter/raw/main/media/terrain.png)

If you find this code useful, please reference in your paper:

```
@misc{hafner2021crafter,
  title = {Benchmarking Diverse Agent Capabilities},
  author = {Danijar Hafner},
  year = {2021},
}
```

## Overview

Crafter features randomly generated 2D worlds where the player needs to forage
for food and water, find shelter to sleep, defend against monsters, collect
materials, and build tools. Crafter aims to be a fruitful benchmark for
reinforcement learning by focusing on the following design goals:

- **Research challenges:** Crafter poses substantial challenges to current
  methods, evaluating strong generalization, wide and deep exploration,
  representation learning, and long-term reasoning and credit assignment.

- **Meaningful evaluation:** Agents are evaluated by semantically meaningful
  achievements that can be unlocked in each episode, offering insights into the
  ability spectrum of both reward agents and unsupervised agents.

- **Iteration speed:** Crafter evaluates many agent abilities within a single
  environment, vastly reducing the computational requirements over benchmarks
  suites that require training on many separate environments from scratch.

## Play Yourself

```sh
python3 -m pip install crafter  # Install Crafter
python3 -m pip install pygame   # Needed for human interface
python3 -m crafter.run_gui      # Start the game
```

<details>
<summary>Keyboard mapping (click to expand)</summary>

| Key | Action |
| :-: | :----- |
| WASD | Move around. |
| SPACE| Collect material, drink from lake, hit creature |
| T | Place a table. |
| R | Place a rock. |
| F | Place a furnace. |
| P | Place a plant. |
| 1 | Craft a wood pickaxe. |
| 2 | Craft a stone pickaxe. |
| 3 | Craft an iron pickaxe. |
| 4 | Craft a wood sword. |
| 5 | Craft a stone sword. |
| 6 | Craft an iron sword. |

</details>

![Crafter Video](https://github.com/danijar/crafter/raw/main/media/video.gif)

## Interface

To install Crafter, run `pip3 install crafter`. The environment follows the
[OpenAI Gym][gym] interface. Observations are images of size (64, 64, 3) and
outputs are one of 17 categorical actions.

```py
import gym
import crafter

env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)

obs = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

[gym]: https://github.com/openai/gym

## Evaluation

The environmnent defines `CrafterReward-v1` for agents that learn from the
provided reward and `CrafterNoReward-v1` for unsupervised agents. Agents are
allowed a budget of 1M environmnent steps and are evaluated by their success
rates on the 22 achievements and by their geometric mean score. Example scripts
for computing these are included in the `analysis` directory of the repository.

- **Reward:** The sparse reward is `+1` for unlocking a new achievement during
  the episode and `-0.1` or `+0.1` for every lost or regenerated health point.
  Performance should not be reported as reward but as the score; see below.

- **Success rates:** The success rates of the 22 achievemnts are computed
  as the percentage across all training episodes in which the achievement was
  unlocked, allowing insights into the ability spectrum of an agent.

- **Crafter score:** The score is the geometric mean of success rates, so that
  improvements on difficult achievements contribute more than improvements on
  achievements with already high success rates. Please see the paper for
  details.

## Baselines

Baseline scores of various agents are available for Crafter, both with and
without rewards. The scores are available in JSON format in the `scores`
directory of the repository. For comparison, the score of human expert players
is 50.5\%. The [baseline
implementations](https://github.com/danijar/crafter-baselines) are available as
a separate repository.

<img src="https://github.com/danijar/crafter/raw/main/media/scores.png" width="400"/>

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/crafter/issues
