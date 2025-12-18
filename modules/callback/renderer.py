import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from myosuite.utils import gym


class PeriodicVideoRecorder(BaseCallback):
  """Records short rollout videos every N environment steps.

  To avoid circular imports, provide `wrap_env_fn`, e.g. `lambda env: ActionSpaceWrapper(env)`.
  """

  def __init__(
      self,
      video_dir: str,
      env_id: Optional[str] = None,
      record_every_steps: int = 50000,
      rollout_steps: int = 500,
      wrap_env_fn: Optional[Callable] = None,
      make_env_fn: Optional[Callable] = None,
      verbose: int = 0
  ):
    super().__init__(verbose)
    self.video_dir = video_dir
    self.env_id = env_id
    self.record_every_steps = record_every_steps
    self.rollout_steps = rollout_steps
    self.wrap_env_fn = wrap_env_fn
    self.make_env_fn = make_env_fn
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
    if self.make_env_fn is not None:
      base_env = self.make_env_fn()
    else:
      try:
        base_env = gym.make(self.env_id) if self.env_id is not None else None
      except TypeError:
        # Some envs may not accept render_mode; fallback to default
        base_env = gym.make(self.env_id) if self.env_id is not None else None

    if base_env is None:
      if self.verbose:
        print("[Video] Unable to determine env_id or make_env_fn for recording; skipping video.")
      return

    env = self.wrap_env_fn(base_env) if self.wrap_env_fn is not None else base_env

    # Use a simple policy rollout for a fixed number of steps
    frames_cam1: List[np.ndarray] = []
    frames_cam2: List[np.ndarray] = []

    obs = env.reset()
    if isinstance(obs, tuple):
      obs = obs[0]

    done = False
    truncated = False
    steps = 0
    while not done and not truncated and steps < self.rollout_steps:
      # Retrieve obs vector if env returns dict
      if isinstance(obs, dict):
        if hasattr(env.unwrapped, 'obsdict2obsvec'):
          obs_vec = env.unwrapped.obsdict2obsvec(
              env.unwrapped.obs_dict, env.unwrapped.obs_keys)[1]
        else:
          obs_vec = obs
      else:
        obs_vec = obs
      # If the model was trained with VecNormalize, normalize observations
      # using the statistics from the training env.
      obs_input = obs_vec
      try:
        model_env = self.model.get_env()
        if isinstance(model_env, VecNormalize):
          obs_input = model_env.normalize_obs(obs_vec)
      except Exception:
        pass

      action, _ = self.model.predict(obs_input, deterministic=True)
      obs, reward, done, truncated, info = env.step(action)

      try:
        # Preferred path: use MuJoCo offscreen renderer if available
        # Check for sim and renderer
        if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "renderer") and hasattr(env.unwrapped.sim.renderer, "render_offscreen"):
          f1 = env.unwrapped.sim.renderer.render_offscreen(
              width=640, height=480, camera_id=1)
          f2 = env.unwrapped.sim.renderer.render_offscreen(
              width=640, height=480, camera_id=2)
          if f1 is not None:
            frames_cam1.append(f1.astype(np.uint8))
          if f2 is not None:
            frames_cam2.append(f2.astype(np.uint8))
        else:
          # Fallback: use Gymnasium render() API
          frame = env.render()
          if frame is None:
            frame = env.render(mode='rgb_array')

          if frame is not None:
            frames_cam1.append(frame.astype(np.uint8))
      except Exception as e:
        if self.verbose:
          print(f"[Video] Rendering failed: {e}")
        break
      steps += 1

    env.close()

    if len(frames_cam1) == 0 and len(frames_cam2) == 0:
      if self.verbose:
        print("[Video] No frames captured; skipping save.")
      return

    # Save mp4 using imageio-ffmpeg (via moviepy or imageio)
    import imageio

    def _save_video(frames, cam_suffix) -> Optional[str]:
      if not frames:
        return None
      video_path = os.path.join(
          self.video_dir,
          f"rollout_{int(time.time())}_steps{self.num_timesteps}{cam_suffix}.mp4",
      )
      imageio.mimwrite(video_path, frames, fps=30, macro_block_size=None)
      return video_path

    # Save videos in parallel (encoding/I/O), then log serially (W&B thread-safety).
    saved: List[tuple[str, str]] = []
    tasks = [(frames_cam1, "_cam1"), (frames_cam2, "_cam2")]
    with ThreadPoolExecutor(max_workers=2) as ex:
      fut_to_suffix = {ex.submit(_save_video, frames, suffix): suffix for frames, suffix in tasks}
      for fut in as_completed(fut_to_suffix):
        suffix = fut_to_suffix[fut]
        try:
          path = fut.result()
          if path is not None:
            saved.append((suffix, path))
            if self.verbose:
              print(f"[Video] Saved {path}")
        except Exception as e:
          print(f"Failed to save video {suffix}: {e}")

    if wandb.run is not None:
      for suffix, path in saved:
        wandb.log(
            {f"rollout/video{suffix}": wandb.Video(path, format="mp4")},
            step=self.num_timesteps,
        )
