import os
# CRITICAL: Must be set before any MuJoCo or MyoSuite imports
os.environ["MUJOCO_GL"] = "egl"

import warnings
warnings.filterwarnings("ignore")

from myosuite.utils import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import numpy as np
import imageio
from typing import Any, Dict, Iterable, Optional, Tuple
from pathlib import Path
import time
import subprocess
import shutil
import argparse
import joblib
import torch
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor

# Internal imports
from SAR.SynergyWrapper import SynNoSynWrapper
from modules.models.hierarchical import HierarchicalTableTennisWrapper
from modules.models.lattice import LatticeActorCriticPolicy
from modules.envs.curriculum import tabletennis_curriculum_kwargs

from modules.callback.checkpoint import resolve_checkpoint_path


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


def get_env_kwargs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
  kwargs = tabletennis_curriculum_kwargs(args.difficulty, reward_type=args.reward_type)

  alignment_weights = None
  if "weighted_reward_keys" in kwargs:
    wk = kwargs["weighted_reward_keys"]
    alignment_weights = {
        "alignment_y": wk.get("alignment_y", 0.5),
        "alignment_z": wk.get("alignment_z", 0.5),
        "paddle_quat_goal": wk.get("paddle_quat_goal", 0.5),
        "pelvis_alignment": wk.get("pelvis_alignment", 0.0),
    }
    wk.pop("alignment_y", None)
    wk.pop("alignment_z", None)
    wk.pop("paddle_quat_goal", None)
    wk.pop("pelvis_alignment", None)

  return kwargs, alignment_weights


def load_sar_artifacts(sar_dir: str, phi: float) -> Dict[str, Any]:
  print(f"Loading SAR artifacts from {sar_dir}...")
  artifacts = {
      "ica": joblib.load(os.path.join(sar_dir, "ica.pkl")),
      "pca": joblib.load(os.path.join(sar_dir, "pca.pkl")),
      "scaler": joblib.load(os.path.join(sar_dir, "scaler.pkl")),
      "phi": phi
  }
  return artifacts


def prepare_env(args: argparse.Namespace, sar_artifacts: Optional[Dict] = None) -> Any:
  """Builds the table tennis env with tuning knobs and wrappers matching evaluator.py."""
  kwargs, alignment_weights = get_env_kwargs(args)
  if args.render_mode:
    kwargs["render_mode"] = args.render_mode

  try:
    env = gym.make(args.env_id, **kwargs)
  except TypeError:
    env = gym.make(args.env_id)

  # Apply wrappers in the same order as evaluator.py
  if args.use_hierarchical:
    env = HierarchicalTableTennisWrapper(
        env, update_freq=args.update_freq, alignment_weights=alignment_weights)

  if args.use_sarl and sar_artifacts:
    env = SynNoSynWrapper(
        env,
        sar_artifacts["ica"],
        sar_artifacts["pca"],
        sar_artifacts["scaler"],
        sar_artifacts["phi"]
    )

  return Monitor(env)


from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv


def capture_frame(sim: Any, width: int, height: int, camera_id: int) -> Optional[np.ndarray]:
  """Captures a frame using the provided simulation object."""
  try:
    return sim.renderer.render_offscreen(
        width=width, height=height, camera_id=camera_id
    ).astype(np.uint8)
  except Exception:
    return None


def rollout(env: Any, model: Any, max_steps: int, deterministic: bool, render_opts: Dict[str, Any]) -> Dict[int, list[np.ndarray]]:
  camera_ids = render_opts["camera_ids"]
  render_every = render_opts.get("render_every", 1)
  frames: Dict[int, list[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}

  is_vectorized = isinstance(env, VecNormalize) or isinstance(env, VecEnv)

  # Optimization: Extract the underlying sim object ONCE per rollout
  curr_env = env
  while hasattr(curr_env, "venv"):
    curr_env = curr_env.venv
  if hasattr(curr_env, "envs"):
    curr_env = curr_env.envs[0]
  sim = getattr(curr_env.unwrapped, "sim", None)

  # Ensure renderer is initialized and optimized
  if sim is not None:
    try:
      # Pre-run forward to initialize scene
      sim.forward()
      # Disable expensive visual features for speed
      sim.model.vis.global_.shadowsize = 0
      sim.model.vis.global_.offsamples = 0
    except Exception:
      pass

  observation = env.reset()
  obs = observation if is_vectorized else (observation[0] if isinstance(observation, tuple) else observation)

  # Support for Recurrent policies
  states = None
  num_envs = env.num_envs if is_vectorized else 1
  episode_start = np.ones((num_envs,), dtype=bool)

  steps = 0
  while steps < max_steps:
    action, states = model.predict(
        obs,
        state=states,
        episode_start=episode_start,
        deterministic=deterministic
    )
    episode_start.fill(False)

    if is_vectorized:
      obs, rewards, dones, infos = env.step(action)
      done = dones[0]
      truncated = infos[0].get("TimeLimit.truncated", False)
    else:
      step = env.step(action)
      if len(step) == 5:
        obs, _, done, truncated, _ = step
      else:
        obs, _, done, _ = step
        truncated = False

    # Capture frames
    if steps % render_every == 0 and sim is not None:
      for cam_id in camera_ids:
        frame = capture_frame(sim, render_opts["width"], render_opts["height"], cam_id)
        if frame is not None:
          frames[cam_id].append(frame)

    steps += 1
    if done or truncated:
      episode_start.fill(True)
      break

  return frames


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Sample PPO/RecurrentPPO checkpoint rollouts and save videos")
  parser.add_argument("--env-id", type=str,
                      default="myoChallengeTableTennisP1-v0", help="Environment ID to render")
  parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to PPO/RecurrentPPO checkpoint zip file or directory")
  parser.add_argument("--output-dir", type=str, default="sample_videos",
                      help="Directory where MP4 files will be written")
  parser.add_argument("--episodes", type=int, default=1,
                      help="Number of rollouts to record")
  parser.add_argument("--max-steps", type=int, default=500,
                      help="Maximum steps per rollout")
  parser.add_argument("--deterministic", action="store_true",
                      help="Use deterministic policy actions")
  parser.add_argument("--render-mode", type=str, default="rgb_array",
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
  parser.add_argument("--render-every", type=int, default=1,
                      help="Render a frame every N steps (e.g., 2 for 25fps if env is 50hz)")

  # Evaluation Arguments (from evaluator.py)
  parser.add_argument("--difficulty", type=int, default=1,
                      help="Curriculum difficulty level (0-5)")
  parser.add_argument("--reward-type", type=str,
                      default="standard", help="Reward type (small/standard)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")

  # Feature Toggles (must match how the model was trained)
  parser.add_argument("--use-sarl", action="store_true",
                      help="Use Synergy Action Reformulation (SARL)")
  parser.add_argument("--use-hierarchical", action="store_true",
                      help="Use Hierarchical wrapper")
  parser.add_argument("--use-lattice", action="store_true",
                      help="Use LatticeActorCriticPolicy")
  parser.add_argument("--use-lstm", action="store_true",
                      help="Use RecurrentPPO (LSTM)")

  # SARL Specific
  parser.add_argument("--sar-dir", type=str, default="SAR",
                      help="Directory containing SAR artifacts")
  parser.add_argument("--phi", type=float, default=0.8,
                      help="Synergy blending parameter (SARL)")

  # Hierarchical Specific
  parser.add_argument("--update-freq", type=int, default=10,
                      help="Goal update frequency in Hierarchical wrapper")

  # Normalization Arguments
  parser.add_argument("--norm-obs", action="store_true",
                      default=True, help="Normalize observations (usually True)")
  parser.add_argument("--clip-obs", type=float, default=10.0,
                      help="Clipping value for observations")

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  # Validation logic
  if args.use_lstm and args.use_lattice:
    raise ValueError(
        "LatticeActorCriticPolicy is not compatible with RecurrentPPO (LSTM).")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  checkpoint_path = resolve_checkpoint_path(args.checkpoint)
  checkpoint_stem = Path(checkpoint_path).stem
  os.makedirs(args.output_dir, exist_ok=True)

  sar_artifacts = load_sar_artifacts(
      args.sar_dir, args.phi) if args.use_sarl else None

  # Prepare base env for model loading/initialization
  def make_env_fn():
    return prepare_env(args, sar_artifacts)

  env = DummyVecEnv([make_env_fn])

  # Handle VecNormalize
  model_path_obj = Path(checkpoint_path)
  vecnorm_path = model_path_obj.parent / "vecnormalize.pkl"
  if not vecnorm_path.exists():
    vecnorm_path = model_path_obj.parent.parent / "vecnormalize.pkl"

  if args.norm_obs:
    if vecnorm_path.exists():
      print(f"Loading VecNormalize statistics from {vecnorm_path}")
      env = VecNormalize.load(str(vecnorm_path), env)
      env.training = False
      env.norm_reward = False
    else:
      print(
          f"Warning: --norm-obs is True but {vecnorm_path} not found. Evaluation might be inaccurate.")

  # Load Model
  print(f"Loading model from {checkpoint_path}...")
  algo_class = RecurrentPPO if args.use_lstm else PPO

  custom_objects = {}
  if args.use_lattice:
    custom_objects["policy_class"] = LatticeActorCriticPolicy

  model = algo_class.load(str(checkpoint_path), env=env,
                          device=device, custom_objects=custom_objects)

  render_opts = {
      "width": args.render_width,
      "height": args.render_height,
      "camera_ids": args.render_camera_ids,
      "render_every": args.render_every,
  }

  for episode in range(1, args.episodes + 1):
    print(f"Rolling out episode {episode}...")
    # We reuse the same env if possible, or reset it.
    # rollout handles env.reset()
    frames_by_cam = rollout(
        env, model, args.max_steps, args.deterministic, render_opts)

    if not any(frames_by_cam.values()):
      print(f"Episode {episode} produced no frames; skipping.")
      continue

    timestamp = int(time.time())
    for cam_id, frames in frames_by_cam.items():
      if not frames:
        continue
      video_name = f"sample_{checkpoint_stem}_cam{cam_id}_ep{episode}_{timestamp}.mp4"
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

  env.close()


if __name__ == "__main__":
  main()
