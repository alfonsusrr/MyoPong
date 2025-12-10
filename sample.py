from myosuite.utils import gym
from stable_baselines3 import PPO
import numpy as np
import imageio
from typing import Any, Dict, Iterable, Optional
from pathlib import Path
import time
import subprocess
import shutil
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")


def _write_with_ffmpeg(frames: Iterable[np.ndarray], path: str, fps: int) -> bool:
  ffmpeg_path = shutil.which("ffmpeg")
  if ffmpeg_path is None:
    print("ffmpeg binary not found; cannot fall back to ffmpeg writer.")
    return False

  frames_list = list(frames)
  if not frames_list:
    return False

  first_frame = frames_list[0]
  height, width = first_frame.shape[:2]
  try:
    proc = subprocess.Popen(
        [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for frame in frames_list:
      proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
      print(proc.stderr.read().decode())
      return False
    return True
  except Exception as exc:
    print("ffmpeg fallback failed:", exc)
    return False


def resolve_checkpoint_path(checkpoint_target: str) -> Path:
  target = Path(checkpoint_target).expanduser()
  if target.is_dir():
    checkpoints = sorted(target.glob("*.zip"))
    if not checkpoints:
      raise FileNotFoundError(f"No checkpoint archives found in {target}")
    return checkpoints[-1].resolve()

  if target.is_file():
    return target.resolve()

  if target.suffix != ".zip":
    zipped_candidate = target.with_suffix(".zip")
    if zipped_candidate.is_file():
      return zipped_candidate.resolve()

  raise FileNotFoundError(f"Checkpoint path {checkpoint_target} does not exist")


def prepare_env(env_id: str, render_mode: Optional[str]) -> Any:
  """Builds the table tennis env with tuning knobs exposed in TableTennisEnvV0.

  The kwargs align with the configuration in `lib/myosuite/envs/myo/myochallenge/tabletennis_v0.py`:
    * `ball_xyz_range`: dict with `low`/`high` samples the ball starting position.
    * `ball_qvel`: bool that enables physics-based velocity sampling to land the ball
      on the agent's side. Requires `ball_xyz_range` to be set as well.
    * `paddle_mass_range`: tuple `(low, high)` for randomized paddle mass dynamics.
    * `frame_skip`: how many physics steps each action persists (default 10 in the env).
    * `rally_count`: number of successful solves before the episode is forced done.
    * `qpos_noise_range`: optional `(low, high)` noise applied to non-ball/paddle joints.
    * `ball_friction_range`: optional dict sets random friction coefficients for the ball geo.
    * `weighted_reward_keys`: map to scale the default reward terms (`reach_dist`,
      `palm_dist`, `paddle_quat`, `act_reg`, `sparse`, `solved`, `done`, etc.).
  """
  # kwargs = {
  #   "ball_xyz_range": {"low": [0.45, -0.35, 1.32], "high": [0.65, 0.35, 1.38]},
  #   "ball_qvel": True,
  #   "paddle_mass_range": (0.6, 1.1),
  #   # "frame_skip": 8,
  #   # "rally_count": 2,
  #   "qpos_noise_range": {"low": 0, "high": 0},
  #   "ball_friction_range": {"low": 0.3, "high": 0.3},
  #   "weighted_reward_keys": {
  #     "reach_dist": 0.6,
  #     "palm_dist": 0.6,
  #     "paddle_quat": 1.5,
  #     "act_reg": 0.5,
  #     "sparse": 50,
  #     "solved": 1200,
  #     "done": -10,
  #   },
  # }

  kwargs = {
      # "ball_xyz_range": {"low": [-1, -0.2, 1.32], "high": [-1, 0.2, 1.40]},
      # "ball_qvel": True,
      # "frame_skip": 8,
      # "paddle_mass_range": (0.60, 0.60),
  }

  if render_mode:
    kwargs["render_mode"] = render_mode
  try:
    env = gym.make(env_id, **kwargs)
  except TypeError:
    env = gym.make(env_id)
  return env


def obs_to_vector(env: Any, obs: Any, model: PPO) -> Any:
  if isinstance(obs, dict):
    if hasattr(env.unwrapped, "obsdict2obsvec"):
      _, obs_vec = env.unwrapped.obsdict2obsvec(
          env.unwrapped.obs_dict, env.unwrapped.obs_keys)
      return obs_vec
    return model.policy.obs_to_tensor(obs)[0].cpu().numpy()
  return obs


def capture_frame(env: Any, width: int, height: int, camera_id: Optional[int]) -> Optional[np.ndarray]:
  try:
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "renderer") and hasattr(sim.renderer, "render_offscreen"):
      return sim.renderer.render_offscreen(width=width, height=height, camera_id=camera_id or 0).astype(np.uint8)
  except Exception:
    pass

  if os.environ.get("DISPLAY") is None:
    # If we are headless, avoid invoking GLFW-backed renderers.
    return None

  try:
    frame = env.render()
  except Exception:
    return None
  if frame is None:
    return None
  return np.asarray(frame, dtype=np.uint8)


def rollout(env: Any, model: PPO, max_steps: int, deterministic: bool, render_opts: Dict[str, Any]) -> Dict[int, list[np.ndarray]]:
  camera_ids = render_opts["camera_ids"]
  frames: Dict[int, list[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}
  observation = env.reset()
  if isinstance(observation, tuple):
    obs, _ = observation
  else:
    obs = observation
  done = False
  truncated = False
  steps = 0
  while not done and not truncated and steps < max_steps:
    obs_vec = obs_to_vector(env, obs, model)
    action, _ = model.predict(obs_vec, deterministic=deterministic)
    step = env.step(action)
    if len(step) == 5:
      obs, _, done, truncated, _ = step
    else:
      obs, _, done, _ = step
      truncated = False
    for cam_id in camera_ids:
      frame = capture_frame(
          env, width=render_opts["width"], height=render_opts["height"], camera_id=cam_id)
      if frame is not None:
        frames[cam_id].append(frame)
    steps += 1
  env.close()
  return frames


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Sample PPO checkpoint rollouts and save videos")
  parser.add_argument("--env-id", type=str,
                      default="myoChallengeTableTennisP1-v0", help="Environment ID to render")
  parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to PPO checkpoint zip file or directory")
  parser.add_argument("--output-dir", type=str, default="sample_videos",
                      help="Directory where MP4 files will be written")
  parser.add_argument("--episodes", type=int, default=1,
                      help="Number of rollouts to record")
  parser.add_argument("--max-steps", type=int, default=500,
                      help="Maximum steps per rollout")
  parser.add_argument("--deterministic", action="store_true",
                      help="Use deterministic policy actions")
  parser.add_argument("--render-mode", type=str, default=None,
                      help="Optional gym render_mode parameter")
  parser.add_argument("--render-width", type=int, default=640,
                      help="Rendered frame width")
  parser.add_argument("--render-height", type=int, default=480,
                      help="Rendered frame height")
  parser.add_argument(
      "--render-camera-ids",
      type=int,
      nargs="+",
      default=[1, 2],
      help="Camera ids to query from the renderer (can pass multiple values)",
  )
  parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint_path = resolve_checkpoint_path(args.checkpoint)
  os.makedirs(args.output_dir, exist_ok=True)
  env = prepare_env(args.env_id, args.render_mode)
  model = PPO.load(str(checkpoint_path), env=env)

  render_opts = {
      "width": args.render_width,
      "height": args.render_height,
      "camera_ids": args.render_camera_ids,
  }

  for episode in range(1, args.episodes + 1):
    env = prepare_env(args.env_id, args.render_mode)
    model.set_env(env)
    frames_by_cam = rollout(env, model, args.max_steps, args.deterministic, render_opts)
    if not any(frames_by_cam.values()):
      print(f"Episode {episode} produced no frames; skipping.")
      continue
    timestamp = int(time.time())
    for cam_id, frames in frames_by_cam.items():
      if not frames:
        continue
      video_name = f"sample_{checkpoint_path.stem}_cam{cam_id}_ep{episode}_{timestamp}.mp4"
      video_path = os.path.join(args.output_dir, video_name)
      try:
        imageio.mimwrite(video_path, frames, fps=args.fps, macro_block_size=None)
      except ValueError:
        if _write_with_ffmpeg(frames, video_path, args.fps):
          print(
              f"Wrote episode {episode} cam{cam_id} via ffmpeg fallback -> {video_path}")
          continue
        raise
      print(f"Wrote episode {episode} cam{cam_id} to {video_path}")


if __name__ == "__main__":
  main()
