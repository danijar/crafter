import datetime
import pathlib

import imageio
import numpy as np
from PIL import Image


class Recorder:

  def __init__(self):
    self._episode = None

  def reset(self, obs):
    # The first time step only contains the initial image that the environment
    # returns on reset.
    self._episode = [{'image': obs}]

  def step(self, action, obs, reward, done, info):
    # Each time step contains the action and the quantities provided by the
    # environment in response to the action.
    info = info.copy()
    info.update({
        f'inventory_{k}': v for k, v
        in info.pop('inventory').items()})
    info.update({
        f'achivement_{k}': v for k, v
        in info.pop('achievements').items()})
    self._episode.append({
        'action': action,
        'image': obs,
        'reward': reward,
        'done': done,
        **info,
    })

  def episode(self):
    # Fill in keys for the first time step of the episode.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    return {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}

  def video(self, size=(256, 256)):
    frames = [step['image'] for step in self._episode]
    frames = [
        np.array(Image.fromarray(frame).resize(size, Image.NEAREST))
        for frame in frames]
    return frames

  def save(self, directory):
    directory = pathlib.Path(directory)
    directory.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    path = str(directory / timestamp)
    imageio.mimsave(path + '.mp4', self.video())
    print('Saved', path + '.mp4')
    np.savez_compressed(path + '.npz', **self.episode())
    print('Saved', path + '.npz')
