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

Crafter is an open world survival game for reinforcement learning research. It
features randomly generated 2D worlds with forests, lakes, mountains, and
caves. The player needs to forage for food and water, find shelter to sleep,
defend against monsters, collect materials, and build tools. The game mechanics
are inspired by the popular game Minecraft and were simplified and optimized
for research productivity. Crafter aims to be a fruitful benchmark for
reinforcement learning by focusing on the following design goals:

- **Research challenges:** Crafter poses substantial challenges to current
  methods. Procedural generation requires strong generalization, the technology
  tree evaluates wide and deep exploration, image observations calls for
  representation learning, repeated subtasks and sparse rewards evaluate
  long-term reasoning and credit assignment.

- **Meaningful evaluation:** Agents are evaluated by a range of achievements
  that can be unlocked in each episode. The achievements correspond to
  meaningful milestones in behavior, offering insights into ability spectrum of
  both reward agents and unsupervised agents.

- **Iteration speed:** Crafter evaluates many agent abilities within a single
  environment, vastly reducing the computational requirements over benchmarks
  suites that require training on many separate environments from scratch,
  while making it more likely that the measured performance is representative
  of new domains.

## Play Yourself

Crafter comes with an optional interface for humans to play the game:

```sh
python3 -m pip install crafter  # Install Crafter
python3 -m pip install pygame   # Needed for human interface
python3 -m crafter.run_gui      # Start the game
```

Command line options are available for recording the games (`--record
directory`), changing the window size (`--window 600 600`), and pausing the
game between key presses (`--wait True`).

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

## Training Agents

The environment follows the [OpenAI Gym][gym] interface. Agent inputs are RGB
images of size 64x64x3 and outputs are one of 17 categorical actions.

Installation: `pip3 install crafter`

```py
import gym
import crafter

env = gym.make('CrafterReward-v1')

assert env.observation_space.shape == (64, 64, 3)
assert env.action_space.n == 17

# Optionally, this recorder makes it easy to compute scores.
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

## Evaluation Protocol

The environmnent defines two benchmarks, one with provided reward signal
(`CrafterReward-v1`) and one for unsupervised agents (`CrafterNoReward-v1`). In
both cases, agents are allowed to interact with the environmnent for 1M
environmnent steps and are evaluated by their success rates on the 22
achievements and by their Crafter score. Example scripts for evaluating agents
are included in the `analysis` directory of the repository.

- **Reward:** Crafter provides a sparse reward signal for reinforcement
    learning agents. A reward of +1 is given every time an achievemnt is
    unlocked for the first time during the episode, and a reward of -0.1 or
    +0.1 is given for every lost or generated health point. The reward should
    only be used for learning, but not to report performance.

- **Success rates:** The success rates of the 22 achievemnts are computed
    as the percentage of episodes during which the achievement was unlocked at
    least once. It is computed over all training episodes within the budget 1M
    environment steps. The achievements cover a range of difficulties and
    challenges, allowing insights into the ability spectrum of an agent.

- **Crafter score:** The score is computed as the mean in log space, also
    known as geometric mean, of the success rates. This ensures that
    improvements on more difficult achievements with lower success rates
    contribute more strongly to the final score than improvements on easy
    achievements with already high success rates.

![Crafter Achievements](https://github.com/danijar/crafter/raw/main/media/achievements.png)

## Baselines

We provide baseline scores of several agents on Crafter, both with reward and
without reward. The scores are available in JSON format in the `scores`
directory of the repository. For comparison, the score of human expert players
is 50.5\%.

Crafter Scores:

<img src="https://github.com/danijar/crafter/raw/main/media/scores.png" width="400"/>

Success Rates with Rewards:

<img src="https://github.com/danijar/crafter/raw/main/media/bars-reward.png"/>

Success Rates without Rewards:

<img src="https://github.com/danijar/crafter/raw/main/media/bars-noreward.png"/>

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/crafter/issues
