import argparse
import os
from typing import Callable

import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from modules.callback.renderer import PeriodicVideoRecorder
from modules.callback.wandb import WandbCallback
from myosuite.utils import gym

# supress warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


def make_env(env_id: str, seed: int, log_dir: str) -> Callable[[], Monitor]:
  def _init():
    env = gym.make(env_id)
    env.seed(seed)
    return Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}.csv"))
  return _init


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Simple PPO trainer for Table Tennis")
  parser.add_argument("--env-id", type=str, default="myoChallengeTableTennisP1-v0", help="Gymnasium environment ID")
  parser.add_argument("--total-timesteps", type=int, default=30000000, help="Total PPO training timesteps")
  parser.add_argument("--log-dir", type=str, default=os.path.join("runs", "ppo_tabletennis"), help="Log directory")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
  parser.add_argument("--policy", type=str, default="MlpPolicy", help="Stable-Baselines3 policy")
  parser.add_argument("--tensorboard-log", type=str, default=None, help="Optional tensorboard log directory")
  parser.add_argument("--save-path", type=str, default=None, help="Path to save the trained model")
  parser.add_argument("--checkpoint-freq", type=int, default=5000000, help="How many steps between checkpoints")
  parser.add_argument("--wandb-project", type=str, default="myosuite-ppo", help="Optional W&B project to log metrics to")
  parser.add_argument("--render-steps", type=int, default=0, help="Record a video every this many timesteps (0 disables)")
  parser.add_argument("--rollout-steps", type=int, default=500, help="Steps per saved rollout")
  parser.add_argument("--lstm-hidden-size", type=int, default=256, help="Hidden state size for LSTM policies")
  parser.add_argument("--n-lstm-layers", type=int, default=1, help="Number of stacked LSTM layers")
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
    make_env(env_id=args.env_id, seed=args.seed + idx, log_dir=log_dir)
    for idx in range(args.num_envs)
  ]

  print(f"Making {len(make_env_fns)} environments")

  vec_env = VecMonitor(DummyVecEnv(make_env_fns))

  is_lstm_policy = "lstm" in args.policy.lower()
  policy_kwargs = (
    {
      "lstm_hidden_size": args.lstm_hidden_size,
      "n_lstm_layers": args.n_lstm_layers,
    }
    if is_lstm_policy
    else None
  )

  model = PPO(
    policy=args.policy,
    env=vec_env,
    verbose=1,
    seed=args.seed,
    tensorboard_log=os.path.abspath(args.tensorboard_log) if args.tensorboard_log else None,
    policy_kwargs=policy_kwargs,
  )

  callbacks = [
    CheckpointCallback(
      save_freq=args.checkpoint_freq,
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
      )
    )

  try:
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
  finally:
    save_path = args.save_path or os.path.join(log_dir, "ppo_tabletennis")
    model.save(save_path)
    vec_env.close()
    print(f"Model saved to: {save_path}")
    if wandb_module is not None:
      wandb_module.finish()


if __name__ == "__main__":
  main()

