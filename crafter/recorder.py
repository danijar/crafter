import datetime
import json
import pathlib

import imageio
import numpy as np


class Recorder:

  def __init__(
      self, env, directory=None, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512)):
    self._env = env
    self._directory = directory and pathlib.Path(directory)
    self._save_stats = save_stats
    self._save_episode = save_episode
    self._save_video = save_video
    self._video_size = video_size
    self._frames = []
    self._episode = None
    if self._directory:
      self._directory.mkdir(exist_ok=True, parents=True)
      if self._save_stats:
        # We keep the file handle open because some cloud storage systems
        # cannot handle quickly re-opening the file many times.
        self._file_handle = (self._directory / 'stats.jsonl').open('a')

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
    self._frames = [self._env.render(self._video_size)]
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Each time step contains the action and the quantities provided by the
    # environment in response to the action.
    obs, reward, done, info = self._env.step(action)
    self._frames.append(self._env.render(self._video_size))
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

  def episode(self):
    # Fill in keys for the first time step of the episode.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    return {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}

  def _save(self):
    eps = self.episode()
    score = round(eps['reward'].sum(), 1)
    length = len(eps['reward'])
    ach = {
        k: v[-1].item() for k, v in eps.items()
        if k.startswith('achievement_')}
    unlocked = sum(int(v > 0) for v in ach.values())
    time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    path = str(self._directory / f'{time}-ach{unlocked:02}-len{length:05}')
    if self._save_stats:
      content = {'score': score, 'length': length}
      content.update(**{
          k: v[-1].item() for k, v in eps.items()
          if k.startswith('achievement_')})
      self._file_handle.write(json.dumps(content) + '\n')
      self._file_handle.flush()
    if self._save_video:
      imageio.mimsave(path + '.mp4', self._frames)
    if self._save_episode:
      np.savez_compressed(path + '.npz', **eps)
