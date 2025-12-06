import os
import time
from typing import List, Optional, Callable

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from myosuite.utils import gym


class PeriodicVideoRecorder(BaseCallback):
  """Records short rollout videos every N environment steps.

  To avoid circular imports, provide `wrap_env_fn`, e.g. `lambda env: ActionSpaceWrapper(env)`.
  """

  def __init__(
    self,
    video_dir: str,
    env_id: str,
    record_every_steps: int = 50000,
    rollout_steps: int = 500,
    wrap_env_fn: Optional[Callable] = None,
    verbose: int = 0
  ):
    super().__init__(verbose)
    self.video_dir = video_dir
    self.env_id = env_id
    self.record_every_steps = record_every_steps
    self.rollout_steps = rollout_steps
    self.wrap_env_fn = wrap_env_fn
    os.makedirs(self.video_dir, exist_ok=True)
    self._last_record_step = 0

  def _on_step(self) -> bool:
    # On the first step after (re)starting or resuming, align to current timestep
    if self._last_record_step == 0 and self.num_timesteps > 0:
      self._last_record_step = self.num_timesteps
    if (self.num_timesteps - self._last_record_step) >= self.record_every_steps:
      self._last_record_step = self.num_timesteps
      self._record_video()
    return True

  def _record_video(self) -> None:
    # Create a fresh env for recording to avoid interfering with training buffers
    try:
      base_env = gym.make(self.env_id) if self.env_id is not None else None
    except TypeError:
      # Some envs may not accept render_mode; fallback to default
      base_env = gym.make(self.env_id) if self.env_id is not None else None
    if base_env is None:
      if self.verbose:
        print("[Video] Unable to determine env_id for recording; skipping video.")
      return

    env = self.wrap_env_fn(base_env) if self.wrap_env_fn is not None else base_env

    # Use a simple policy rollout for a fixed number of steps
    frames: List[np.ndarray] = []
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    while not done and not truncated and steps < self.rollout_steps:
      # Retrieve obs vector if env returns dict
      if isinstance(obs, dict):
        if hasattr(env.unwrapped, 'obsdict2obsvec'):
          obs_vec = env.unwrapped.obsdict2obsvec(env.unwrapped.obs_dict, env.unwrapped.obs_keys)[1]
        else:
          obs_vec = self.model.policy.obs_to_tensor(obs)[0].cpu().numpy()
      else:
        obs_vec = obs

      action, _ = self.model.predict(obs_vec, deterministic=True)
      obs, reward, done, truncated, info = env.step(action)

      try:
        # Preferred path: use MuJoCo offscreen renderer if available
        if hasattr(env.unwrapped.sim, "renderer") and hasattr(env.unwrapped.sim.renderer, "render_offscreen"):
          frame = env.unwrapped.sim.renderer.render_offscreen(width=640, height=480, camera_id=1)
        else:
          # Fallback: use Gymnasium render() API
          frame = env.render()
        if frame is None:
          break
        frames.append(frame.astype(np.uint8))
      except Exception as e:
        if self.verbose:
          print(f"[Video] Rendering failed: {e}")
        break
      steps += 1
    env.close()

    if len(frames) == 0:
      if self.verbose:
        print("[Video] No frames captured; skipping save.")
      return

    # Save mp4 using imageio-ffmpeg (via moviepy or imageio)
    import imageio
    video_path = os.path.join(self.video_dir, f"rollout_{int(time.time())}_steps{self.num_timesteps}.mp4")
    imageio.mimwrite(video_path, frames, fps=30, macro_block_size=None)
    wandb.log({"rollout/video": wandb.Video(video_path, fps=30, format="mp4")}, step=self.num_timesteps)
    if self.verbose:
      print(f"[Video] Saved {video_path}")


