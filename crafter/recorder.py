import datetime
import json
import pathlib

import imageio
import numpy as np


class Recorder:

  def __init__(
      self, env, directory, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512)):
    if directory and save_stats:
      env = StatsRecorder(env, directory)
    if directory and save_video:
      env = VideoRecorder(env, directory, video_size)
    if directory and save_episode:
      env = EpisodeRecorder(env, directory)
    self._env = env

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)


class StatsRecorder:

  def __init__(self, env, directory):
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._file = (self._directory / 'stats.jsonl').open('a')
    self._length = None
    self._reward = None
    self._unlocked = None
    self._stats = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      for key, value in info['achievements'].items():
        self._stats[f'achievement_{key}'] = value
      self._save()
    return obs, reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()


class VideoRecorder:

  def __init__(self, env, directory, size=(512, 512)):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._size = size
    self._frames = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._frames = [self._env.render(self._size)]
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._frames.append(self._env.render(self._size))
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.mp4'))
    imageio.mimsave(filename, self._frames)


class EpisodeRecorder:

  def __init__(self, env, directory):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._episode = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Transitions are defined from the environment perspective, meaning that a
    # transition contains the action and the resulting reward and next
    # observation produced by the environment in response to said action.
    obs, reward, done, info = self._env.step(action)
    transition = {
        'action': action, 'image': obs, 'reward': reward, 'done': done,
    }
    for key, value in info.items():
      if key in ('inventory', 'achievements'):
        continue
      transition[key] = value
    for key, value in info['achievements'].items():
      transition[f'achievement_{key}'] = value
    for key, value in info['inventory'].items():
      transition[f'ainventory_{key}'] = value
    self._episode.append(transition)
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.npz'))
    # Fill in zeros for keys missing at the first time step.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    episode = {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}
    np.savez_compressed(filename, **episode)


class EpisodeName:

  def __init__(self, env):
    self._env = env
    self._timestamp = None
    self._unlocked = None
    self._length = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._timestamp = None
    self._unlocked = None
    self._length = 0
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    if done:
      self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
      self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
    return obs, reward, done, info

  @property
  def episode_name(self):
    return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
