import os
import time
import argparse
import warnings
import joblib
from pathlib import Path
from typing import Any, Callable
from dotenv import load_dotenv

import numpy as np
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from SAR.SynergyWrapper import SynNoSynWrapper
from modules.models.hierarchical import HierarchicalTableTennisWrapper
from modules.callback.progress import TqdmProgressCallback
from modules.callback.evaluator import PeriodicEvaluator
from modules.callback.wandb import WandbCallback
from modules.callback.renderer import PeriodicVideoRecorder
from modules.envs.curriculum import tabletennis_curriculum_kwargs

# suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()

def prepare_env(env_id: str, difficulty: int = 0) -> Any:
  """Builds the table tennis env aligned with MyoChallenge 2025 Phases."""
  kwargs = tabletennis_curriculum_kwargs(difficulty)
  try:
    env = gym.make(env_id, **kwargs)
  except TypeError:
    print(f"Warning: {env_id} did not accept kwargs. Creating default env.")
    env = gym.make(env_id)
  return env

def make_env(
    env_id: str, seed: int, log_dir: str, ica, pca, scaler, phi, difficulty: int = 0
) -> Callable[[], Monitor]:
  def _init():
    env = prepare_env(env_id, difficulty=difficulty)
    env.seed(seed)
    # Apply Hierarchical wrapper first (Observation augmentation)
    env = HierarchicalTableTennisWrapper(env)
    # Then apply SAR wrapper (Action space modification)
    env = SynNoSynWrapper(env, ica, pca, scaler, phi)
    return Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}.csv"))

  return _init

def resolve_checkpoint_path(checkpoint_target: str) -> str:
  target = Path(checkpoint_target).expanduser().resolve()
  if target.is_dir():
    checkpoints = sorted(
        target.glob("*.zip"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
      raise FileNotFoundError(f"No checkpoint archives found in {target}")
    return str(checkpoints[-1])

  if target.is_file():
    return str(target)

  if target.suffix != ".zip":
    zipped_candidate = target.with_suffix(".zip")
    if zipped_candidate.is_file():
      return str(zipped_candidate)

  raise FileNotFoundError(f"Checkpoint path {checkpoint_target} does not exist")

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="PPO Hierarchical SARL trainer for Table Tennis")
  parser.add_argument("--env-id", type=str,
                      default="myoChallengeTableTennisP1-v0", help="Gymnasium environment ID")
  parser.add_argument("--total-timesteps", type=int, default=20000000,
                      help="Total PPO training timesteps")
  parser.add_argument("--log-dir", type=str, default=os.path.join("runs",
                      "ppo_hierarchical_sarl_tabletennis"), help="Log directory")
  parser.add_argument("--sar-dir", type=str, default="SAR",
                      help="Directory containing SAR artifacts (ica.pkl, pca.pkl, scaler.pkl)")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--num-envs", type=int, default=4,
                      help="Number of parallel environments")
  parser.add_argument("--policy", type=str, default="MlpPolicy",
                      help="Stable-Baselines3 policy (MlpPolicy only)")
  parser.add_argument("--tensorboard-log", type=str, default=None,
                      help="Optional tensorboard log directory")
  parser.add_argument("--checkpoint-freq", type=int, default=2500000,
                      help="How many timesteps between checkpoints")
  parser.add_argument("--wandb-project", type=str, default="myosuite-ppo-hierarchical-sarl",
                      help="Optional W&B project to log metrics to")
  parser.add_argument(
      "--save-path",
      type=str,
      default=None,
      help="Optional path to save the final model",
  )
  parser.add_argument("--render-steps", type=int, default=0,
                      help="Record a video every this many timesteps (0 disables)")
  parser.add_argument("--rollout-steps", type=int, default=500,
                      help="Steps per saved rollout")
  parser.add_argument(
      "--resume-from-checkpoint",
      type=str,
      default=None,
      help="Path to a checkpoint to resume training from",
  )
  parser.add_argument("--eval-freq", type=int, default=10000,
                      help="Evaluate policy every N steps")
  parser.add_argument("--eval-envs", type=int, default=2,
                      help="Number of parallel eval envs")
  parser.add_argument("--eval-episodes", type=int, default=10,
                      help="Total eval episodes per evaluation run")
  parser.add_argument("--n-steps", type=int, default=4096,
                      help="Number of steps per environment per update")
  parser.add_argument("--batch-size", type=int, default=2048,
                      help="Size of the batch for training")
  parser.add_argument("--difficulty", type=int, default=2,
                      help="Curriculum difficulty level (0-4)")
  return parser.parse_args()

def main() -> None:
  args = parse_args()

  run_id = f"run-ppo-h-sarl-{time.strftime('%Y%m%d-%H%M%S')}-lvl{args.difficulty}"

  log_dir = os.path.abspath(os.path.join(args.log_dir, run_id))
  os.makedirs(log_dir, exist_ok=True)

  checkpoint_dir = os.path.abspath(os.path.join(log_dir, "checkpoints"))
  os.makedirs(checkpoint_dir, exist_ok=True)
  video_dir = os.path.abspath(os.path.join(log_dir, "videos"))

  # Load SAR artifacts
  print(f"Loading SAR artifacts from {args.sar_dir}...")
  ica = joblib.load(os.path.join(args.sar_dir, "ica.pkl"))
  pca = joblib.load(os.path.join(args.sar_dir, "pca.pkl"))
  scaler = joblib.load(os.path.join(args.sar_dir, "scaler.pkl"))
  phi = 0.8
  print("SAR artifacts loaded.")

  make_env_fns = [
      make_env(
          env_id=args.env_id,
          seed=args.seed + idx,
          log_dir=log_dir,
          ica=ica,
          pca=pca,
          scaler=scaler,
          phi=phi,
          difficulty=args.difficulty
      )
      for idx in range(args.num_envs)
  ]

  print(f"Making {len(make_env_fns)} environments with difficulty level {args.difficulty}")

  # Training envs
  train_env = SubprocVecEnv(make_env_fns)
  train_env = VecMonitor(train_env)
  vec_env = VecNormalize(
      train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0
  )

  # Eval Envs
  make_eval_env_fns = [
      make_env(env_id=args.env_id, seed=args.seed + 12345 + idx, log_dir=log_dir,
               ica=ica, pca=pca, scaler=scaler, phi=phi, difficulty=args.difficulty)
      for idx in range(args.eval_envs)
  ]
  eval_env = SubprocVecEnv(make_eval_env_fns)
  eval_env = VecMonitor(eval_env)
  eval_vec_env = VecNormalize(
      eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0
  )
  eval_vec_env.obs_rms = vec_env.obs_rms

  # Metrics Env (single instance)
  try:
    metrics_env = prepare_env(args.env_id, difficulty=args.difficulty)
  except TypeError:
    metrics_env = gym.make(args.env_id)
  
  # Hierarchical wrapping for consistency
  metrics_env = HierarchicalTableTennisWrapper(metrics_env)
  metrics_env = metrics_env.unwrapped

  checkpoint_save_freq = max(1, args.checkpoint_freq // args.num_envs)
  model = None
  remaining_timesteps = args.total_timesteps

  if args.resume_from_checkpoint:
    checkpoint_path = resolve_checkpoint_path(args.resume_from_checkpoint)
    print(f"Resuming PPO from checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=vec_env)
    current_timesteps = model.num_timesteps
    remaining_timesteps = args.total_timesteps - current_timesteps
    if remaining_timesteps <= 0:
      print(f"Target timesteps reached. Exiting.")
      remaining_timesteps = 0
  else:
    model = PPO(
        policy=args.policy,
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.abspath(args.tensorboard_log) if args.tensorboard_log else None,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(
            log_std_init=-2,
            net_arch=[256, 256]
        ),
    )

  callbacks = [
      CheckpointCallback(
          save_freq=checkpoint_save_freq,
          save_path=checkpoint_dir,
          name_prefix="ppo_h_sarl",
      )
  ]
  
  if args.wandb_project:
    import wandb as _wandb
    _wandb.init(project=args.wandb_project, name=run_id, config=vars(args))
    callbacks.append(WandbCallback())

  callbacks.append(PeriodicEvaluator(
      eval_vec=eval_vec_env,
      eval_freq=args.eval_freq,
      eval_episodes=args.eval_episodes,
      metrics_env=metrics_env,
      verbose=1
  ))

  callbacks.append(TqdmProgressCallback(total_steps=args.total_timesteps))

  if args.render_steps > 0:
    os.makedirs(video_dir, exist_ok=True)
    render_monitor_file = os.path.join(log_dir, "renderer_monitor.csv")
    def _wrap_env_for_rendering(env):
      env = HierarchicalTableTennisWrapper(env)
      env = SynNoSynWrapper(env, ica, pca, scaler, phi)
      return Monitor(env, filename=render_monitor_file)

    callbacks.append(
        PeriodicVideoRecorder(
            video_dir=video_dir,
            env_id=args.env_id,
            record_every_steps=args.render_steps,
            rollout_steps=args.rollout_steps,
            wrap_env_fn=_wrap_env_for_rendering,
            verbose=1,
            make_env_fn=lambda: prepare_env(args.env_id, difficulty=args.difficulty)
        )
    )

  try:
    if remaining_timesteps > 0:
      model.learn(
          total_timesteps=remaining_timesteps,
          callback=callbacks,
          reset_num_timesteps=not bool(args.resume_from_checkpoint),
      )
  finally:
    if isinstance(vec_env, VecNormalize):
      vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    final_save_path = args.save_path or os.path.join(log_dir, "ppo_h_sarl_tabletennis")
    model.save(final_save_path)
    try:
      vec_env.close()
      eval_vec_env.close()
      metrics_env.close()
    except:
      pass
    print(f"Model saved to: {final_save_path}")
    if args.wandb_project:
      _wandb.finish()

if __name__ == "__main__":
  main()

