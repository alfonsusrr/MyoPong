import os
import time
import argparse
from typing import List
from dotenv import load_dotenv

import numpy as np
import imageio
import joblib

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# Use MyoSuite's gym wrapper
from myosuite.utils import gym

# Match training curriculum difficulty knobs
from modules.envs.curriculum import tabletennis_curriculum_kwargs

# Use the same SARL action wrapper as in training
from SAR.SynergyWrapper import SynNoSynWrapper

load_dotenv()


def render_episode(
    model: PPO,
    env: gym.Env,
    max_steps: int = 1000,
    camera_id: int = 1,
    width: int = 640,
    height: int = 480,
) -> List[np.ndarray]:
  """
  Roll out one episode with the trained policy and collect rendered frames.
  """
  frames: List[np.ndarray] = []

  obs, _ = env.reset()
  done = False
  truncated = False
  step = 0

  while not done and not truncated and step < max_steps:
    # Convert dict obs -> vector if needed (MyoSuite often returns dicts)
    if isinstance(obs, dict):
      if hasattr(env.unwrapped, "obsdict2obsvec"):
        # Use env's own obs conversion if available
        obs_vec = env.unwrapped.obsdict2obsvec(
            env.unwrapped.obs_dict, env.unwrapped.obs_keys
        )[1]
      else:
        # Fallback: SB3's internal conversion (less ideal)
        obs_vec = model.policy.obs_to_tensor(obs)[0].cpu().numpy()
    else:
      obs_vec = obs

    # Get deterministic action from policy
    action, _ = model.predict(obs_vec, deterministic=True)

    # Env step (Gymnasium-style)
    obs, reward, done, truncated, info = env.step(action)

    # Render frame
    try:
      # Preferred: MuJoCo offscreen renderer
      if (
          hasattr(env.unwrapped, "sim")
          and hasattr(env.unwrapped.sim, "renderer")
          and hasattr(env.unwrapped.sim.renderer, "render_offscreen")
      ):
        frame = env.unwrapped.sim.renderer.render_offscreen(
            width=width, height=height, camera_id=camera_id
        )
      else:
        # Fallback: env.render()
        frame = env.render()

      if frame is None:
        print(f"Warning: render returned None at step {step}")
        break

      frames.append(frame.astype(np.uint8))

    except Exception as e:
      print(f"Error rendering frame at step {step}: {e}")
      break

    step += 1

  return frames


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
  """
  Save a list of frames (H x W x 3 uint8) as an MP4.
  """
  if len(frames) == 0:
    print("Warning: no frames to save.")
    return

  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  try:
    imageio.mimwrite(output_path, frames, fps=fps, macro_block_size=None)
    print(f"Video saved to: {output_path}")
  except Exception as e:
    print(f"Error saving video to {output_path}: {e}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Render videos from a trained PPO checkpoint on MyoSuite."
  )
  parser.add_argument(
      "--checkpoint-path",
      type=str,
      required=True,
      help="Path to the PPO checkpoint (.zip) to load.",
  )
  parser.add_argument(
      "--algo",
      type=str,
      default="ppo",
      choices=["ppo", "recurrent_ppo"],
      help="Which algorithm checkpoint to load (default: ppo).",
  )
  parser.add_argument(
      "--env-id",
      type=str,
      default="myoChallengeTableTennisP1-v0",
      help="Gym environment ID (default: myoChallengeTableTennisP1-v0).",
  )
  parser.add_argument(
      "--difficulty",
      type=int,
      default=0,
      help="Curriculum difficulty level (0-4). Passed as env kwargs when supported.",
  )
  parser.add_argument(
      "--num-episodes",
      type=int,
      default=5,
      help="Number of episodes to render (default: 5).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for NumPy and env reset (default: 42).",
  )
  parser.add_argument(
      "--max-steps",
      type=int,
      default=1000,
      help="Maximum number of steps per episode (default: 1000).",
  )
  parser.add_argument(
      "--fps",
      type=int,
      default=30,
      help="Frames per second for saved videos (default: 30).",
  )
  parser.add_argument(
      "--width",
      type=int,
      default=640,
      help="Rendered frame width (default: 640).",
  )
  parser.add_argument(
      "--height",
      type=int,
      default=480,
      help="Rendered frame height (default: 480).",
  )
  parser.add_argument(
      "--camera-id",
      type=int,
      default=2,
      help="MuJoCo camera ID to render from (default: 2).",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default="./videos",
      help="Directory where output videos will be saved (default: ./videos).",
  )
  parser.add_argument(
      "--use-sar",
      action="store_true",
      default=False,
      help="If set, wrap the env with SAR SynNoSynWrapper (requires --sar-dir).",
  )
  parser.add_argument(
      "--sar-dir",
      type=str,
      default="SAR",
      help="Directory containing SAR artifacts (ica.pkl, pca.pkl, scaler.pkl).",
  )
  parser.add_argument(
      "--phi",
      type=float,
      default=0.8,
      help="Mixing weight phi used in SynNoSynWrapper (default: 0.8).",
  )
  return parser.parse_args()


def main(args: argparse.Namespace) -> None:
  os.makedirs(args.output_dir, exist_ok=True)

  # Set numpy seed for reproducibility
  np.random.seed(args.seed)

  # Load model
  print(f"Loading model from: {args.checkpoint_path}")
  if args.algo == "ppo":
    model = PPO.load(args.checkpoint_path)
  else:
    # RecurrentPPO checkpoints (e.g. Lattice) may require the policy class to be importable.
    model = RecurrentPPO.load(args.checkpoint_path)
  print("Model loaded successfully.")

  # Create base env and optionally wrap with SAR SynNoSynWrapper
  print(f"Creating environment: {args.env_id} (difficulty={args.difficulty})")
  env_kwargs = tabletennis_curriculum_kwargs(args.difficulty)
  try:
    base_env = gym.make(args.env_id, **env_kwargs)
  except TypeError:
    # Some envs may not accept kwargs; fall back to default creation.
    base_env = gym.make(args.env_id)
  env = base_env

  if args.use_sar:
    print(f"Loading SAR artifacts from {args.sar_dir}...")
    ica = joblib.load(os.path.join(args.sar_dir, "ica.pkl"))
    pca = joblib.load(os.path.join(args.sar_dir, "pca.pkl"))
    scaler = joblib.load(os.path.join(args.sar_dir, "scaler.pkl"))
    phi = float(args.phi)
    print("SAR artifacts loaded.")
    env = SynNoSynWrapper(base_env, ica, pca, scaler, phi)

  checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]

  try:
    for episode in range(args.num_episodes):
      print(f"\n=== Episode {episode + 1}/{args.num_episodes} ===")
      start_time = time.time()

      frames = render_episode(
          model=model,
          env=env,
          max_steps=args.max_steps,
          camera_id=args.camera_id,
          width=args.width,
          height=args.height,
      )

      render_time = time.time() - start_time
      print(f"Rendered {len(frames)} frames in {render_time:.2f} seconds")

      if len(frames) > 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{checkpoint_name}_cam{args.camera_id}_episode_{episode + 1}_{timestamp}.mp4"
        output_path = os.path.join(args.output_dir, filename)
        save_video(frames, output_path, fps=args.fps)
      else:
        print(f"Warning: no frames captured for episode {episode + 1}")
  finally:
    env.close()
    print(f"\nRendering complete. Videos saved to: {args.output_dir}")


if __name__ == "__main__":
  cli_args = parse_args()
  main(cli_args)
