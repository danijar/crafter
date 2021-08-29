import datetime
import pathlib

import imageio
import numpy as np


class Recorder:

  def __init__(
      self, env, directory=None, size=(512, 512),
      save_video=True, save_episode=True, include_image=True):
    self._env = env
    self._directory = directory and pathlib.Path(directory)
    self._size = size
    self._save_episode = save_episode
    self._save_video = save_video
    self._include_image = include_image
    self._frames = []
    self._episode = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def reset(self):
    # The first time step only contains the initial image that the environment
    # returns on reset.
    obs = self._env.reset()
    self._frames = [self._env.render(self._size)]
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Each time step contains the action and the quantities provided by the
    # environment in response to the action.
    obs, reward, done, info = self._env.step(action)
    self._frames.append(self._env.render(self._size))
    details = info.copy()
    details.update({
        f'inventory_{k}': v for k, v
        in details.pop('inventory').items()})
    details.update({
        f'achievement_{k}': v for k, v
        in details.pop('achievements').items()})
    self._episode.append({
        'action': action,
        'image': obs,
        'reward': reward,
        'done': done,
        **details,
    })
    if done and self._directory:
      self._save()
    return obs, reward, done, info

  @property
  def last_frame(self):
    return self._frames[-1]

  def episode(self):
    # Fill in keys for the first time step of the episode.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    return {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}

  def _save(self):
    self._directory.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    path = str(self._directory / timestamp)
    if self._save_video:
      imageio.mimsave(path + '.mp4', self._frames)
      print('Saved', path + '.mp4')
    if self._save_episode:
      episode = self.episode()
      if not self._include_image:
        episode.pop('image')
      np.savez_compressed(path + '.npz', **episode)
      print('Saved', path + '.npz')
