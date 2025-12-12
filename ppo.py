from myosuite.utils import gym
from modules.callback.progress import TqdmProgressCallback
from modules.callback.evaluator import PeriodicEvaluator
from modules.callback.wandb import WandbCallback
from modules.callback.renderer import PeriodicVideoRecorder
from modules.envs.curriculum import tabletennis_curriculum_kwargs
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from pathlib import Path
import time
from typing import Any, Callable
import os
import argparse
import warnings

# supress all warnings
warnings.filterwarnings("ignore")


def prepare_env(env_id: str, difficulty: int = 0) -> Any:
  """Builds the table tennis env aligned with MyoChallenge 2025 Phases."""

  kwargs = tabletennis_curriculum_kwargs(difficulty)

  try:
    env = gym.make(env_id, **kwargs)
  except TypeError:
    print(f"Warning: {env_id} did not accept kwargs. Creating default env.")
    env = gym.make(env_id)

  return env


def make_env(env_id: str, seed: int, log_dir: str, difficulty: int = 0) -> Callable[[], Monitor]:
  def _init():
    env = prepare_env(env_id, difficulty=difficulty)
    env.seed(seed)
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
  parser = argparse.ArgumentParser(description="Simple PPO trainer for Table Tennis")
  parser.add_argument("--env-id", type=str,
                      default="myoChallengeTableTennisP1-v0", help="Gymnasium environment ID")
  parser.add_argument("--total-timesteps", type=int, default=30000000,
                      help="Total PPO training timesteps")
  parser.add_argument("--log-dir", type=str, default=os.path.join("runs",
                      "ppo_tabletennis"), help="Log directory")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--num-envs", type=int, default=2,
                      help="Number of parallel environments")
  parser.add_argument("--policy", type=str, default="MlpPolicy",
                      help="Stable-Baselines3 policy")
  parser.add_argument("--tensorboard-log", type=str, default=None,
                      help="Optional tensorboard log directory")
  parser.add_argument("--save-path", type=str, default=None,
                      help="Path to save the trained model")
  parser.add_argument("--checkpoint-freq", type=int, default=5000000,
                      help="How many steps between checkpoints")
  parser.add_argument("--wandb-project", type=str, default="myosuite-ppo",
                      help="Optional W&B project to log metrics to")
  parser.add_argument("--render-steps", type=int, default=0,
                      help="Record a video every this many timesteps (0 disables)")
  parser.add_argument("--rollout-steps", type=int, default=500,
                      help="Steps per saved rollout")
  parser.add_argument("--lstm-hidden-size", type=int, default=256,
                      help="Hidden state size for LSTM policies")
  parser.add_argument("--n-lstm-layers", type=int, default=1,
                      help="Number of stacked LSTM layers")
  parser.add_argument("--eval-freq", type=int, default=10000,
                      help="Evaluate policy every N steps")
  parser.add_argument("--eval-envs", type=int, default=2,
                      help="Number of parallel eval envs")
  parser.add_argument("--eval-episodes", type=int, default=10,
                      help="Total eval episodes per evaluation run")
  parser.add_argument("--n-steps", type=int, default=4096,
                      help="Number of steps to run for each environment per update")
  parser.add_argument("--batch-size", type=int, default=2048,
                      help="Size of the batch for training")
  parser.add_argument("--difficulty", type=int, default=0,
                      help="Curriculum difficulty level (0-4)")
  parser.add_argument(
      "--resume-from-checkpoint",
      type=str,
      default=None,
      help="Path to a checkpoint (file or directory) to resume training from",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  run_id = f"run-ppo-{time.strftime('%Y%m%d-%H%M%S')}"

  log_dir = os.path.abspath(os.path.join(args.log_dir, run_id))
  os.makedirs(log_dir, exist_ok=True)

  checkpoint_dir = os.path.abspath(os.path.join(log_dir, "checkpoints"))
  os.makedirs(checkpoint_dir, exist_ok=True)
  video_dir = os.path.abspath(os.path.join(log_dir, "videos"))

  make_env_fns = [
      make_env(env_id=args.env_id, seed=args.seed + idx,
               log_dir=log_dir, difficulty=args.difficulty)
      for idx in range(args.num_envs)
  ]

  print(f"Making {len(make_env_fns)} environments with difficulty level {args.difficulty}")

  vec_env = VecMonitor(SubprocVecEnv(make_env_fns))

  make_eval_env_fns = [
      make_env(
          env_id=args.env_id,
          seed=args.seed + 12345 + idx,
          log_dir=log_dir,
          difficulty=args.difficulty,
      )
      for idx in range(args.eval_envs)
  ]
  eval_vec_env = VecMonitor(SubprocVecEnv(make_eval_env_fns))

  # Metrics Env (single instance, unwrapped)
  try:
    metrics_env = prepare_env(args.env_id, difficulty=args.difficulty)
  except TypeError:
    metrics_env = gym.make(args.env_id)
  metrics_env = metrics_env.unwrapped

  is_lstm_policy = "lstm" in args.policy.lower()
  policy_kwargs = dict(
      # gSDE kwargs
      log_std_init=-2,       # Start with moderate noise (not too crazy)
      net_arch=[256, 256]    # Ensure network size is explicit
  )
  if is_lstm_policy:
    policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
    policy_kwargs["n_lstm_layers"] = args.n_lstm_layers

  model = None
  remaining_timesteps = args.total_timesteps

  if args.resume_from_checkpoint:
    checkpoint_path = resolve_checkpoint_path(args.resume_from_checkpoint)
    print(f"Resuming PPO from checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=vec_env)

    current_timesteps = model.num_timesteps
    remaining_timesteps = args.total_timesteps - current_timesteps

    if remaining_timesteps <= 0:
      print(
          f"Model already trained for {current_timesteps} steps (target: {args.total_timesteps}). Exiting training loop.")
      remaining_timesteps = 0
    else:
      print(
          f"Resuming training for {remaining_timesteps} steps (to reach {args.total_timesteps})")
  else:
    model = PPO(
        policy=args.policy,
        env=vec_env,
        n_steps=args.n_steps,           # High batch size for stability
        batch_size=args.batch_size,     # Large mini-batch for M4 GPU/AMX speed
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.abspath(
            args.tensorboard_log) if args.tensorboard_log else None,
        use_sde=True,              # Enable gSDE
        sde_sample_freq=4,         # Resample noise every 4 steps (smooths movement)
        policy_kwargs=policy_kwargs,
    )

  checkpoint_save_freq = max(1, args.checkpoint_freq // args.num_envs)
  checkpoint_timesteps = checkpoint_save_freq * args.num_envs
  print(
      f"Saving checkpoints every {checkpoint_save_freq} env.step() calls (~{checkpoint_timesteps} timesteps)"
  )

  callbacks = [
      CheckpointCallback(
          save_freq=checkpoint_save_freq,
          save_path=checkpoint_dir,
          name_prefix="ppo",
      )
  ]
  wandb_module = None

  if args.wandb_project:
    import wandb as _wandb

    wandb_module = _wandb
    wandb_module.init(
        project=args.wandb_project,
        name=run_id,
        config=vars(args),
    )
    callbacks.append(WandbCallback())

  if args.render_steps > 0:
    os.makedirs(video_dir, exist_ok=True)
    render_monitor_file = os.path.join(log_dir, "renderer_monitor.csv")

    def _wrap_env_for_rendering(env):
      return Monitor(env, filename=render_monitor_file)

    callbacks.append(
        PeriodicVideoRecorder(
            video_dir=video_dir,
            env_id=args.env_id,
            record_every_steps=args.render_steps,
            rollout_steps=args.rollout_steps,
            wrap_env_fn=_wrap_env_for_rendering,
            verbose=1,
            # Use prepare_env so renderer follows curriculum difficulty too
            make_env_fn=lambda: prepare_env(
                args.env_id, difficulty=args.difficulty),
        )
    )

  callbacks.append(PeriodicEvaluator(
      eval_vec=eval_vec_env,
      eval_freq=args.eval_freq,
      eval_episodes=args.eval_episodes,
      metrics_env=metrics_env,
      verbose=1
  ))

  callbacks.append(TqdmProgressCallback(total_steps=args.total_timesteps))

  try:
    if remaining_timesteps > 0:
      model.learn(
          total_timesteps=remaining_timesteps,
          callback=callbacks,
          reset_num_timesteps=not bool(args.resume_from_checkpoint),
      )
    else:
      print("Skipping training as target timesteps reached.")
  finally:
    save_path = args.save_path or os.path.join(log_dir, "ppo_tabletennis")
    model.save(save_path)
    # SubprocVecEnv can raise EOFError on close if a worker died earlier.
    try:
      vec_env.close()
    except EOFError:
      print("Warning: VecEnv worker already terminated (EOFError on close).")

    try:
      eval_vec_env.close()
    except EOFError:
      pass

    try:
      metrics_env.close()
    except Exception:
      pass
    print(f"Model saved to: {save_path}")
    if wandb_module is not None:
      wandb_module.finish()


if __name__ == "__main__":
  main()
