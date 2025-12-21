import os
import time
import argparse
import warnings
from pathlib import Path
from typing import Any, Callable

from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb

from modules.callback.renderer import PeriodicVideoRecorder
from modules.callback.wandb import WandbCallback
from modules.callback.evaluator import PeriodicEvaluator
from modules.callback.progress import TqdmProgressCallback
from modules.envs.curriculum import tabletennis_curriculum_kwargs

# suppress all warnings
warnings.filterwarnings("ignore")


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
    parser = argparse.ArgumentParser(description="PPO trainer for Pong")
    parser.add_argument("--env-id", type=str, default="myoChallengePongP0-v0", help="Environment ID")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--difficulty", type=int, default=3, help="Curriculum difficulty (0-4)")
    parser.add_argument("--log-dir", type=str, default=os.path.join("runs", "pong_ppo_simple"), help="Log directory")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency")
    parser.add_argument("--render-steps", type=int, default=100_000, help="Video recording frequency")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-project", type=str, default="myosuite-ghost-pong", help="W&B project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="W&B run ID to resume")
    return parser.parse_args()


def make_env(env_id: str, seed: int, log_dir: str, difficulty: int) -> Callable[[], Monitor]:
    def _init():
        kwargs = tabletennis_curriculum_kwargs(difficulty=difficulty)
        kwargs.pop('weighted_reward_keys', None)
        env = gym.make(env_id, **kwargs)
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}.csv"))
    return _init


def main():
    args = parse_args()
    seed = args.seed
    
    # 1. Directory Setup
    run_id = f"pong-ppo-{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.abspath(os.path.join(args.log_dir, run_id))
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    video_dir = os.path.join(log_dir, "videos")

    # 2. Wandb Initialization
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": run_id,
        "config": vars(args),
    }
    if args.resume_from_checkpoint and args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
    
    wandb.init(**wandb_kwargs)

    # 3. Environment Setup
    make_env_fns = [
        make_env(env_id=args.env_id, seed=seed + idx, log_dir=log_dir, difficulty=args.difficulty)
        for idx in range(args.num_envs)
    ]
    
    train_env = SubprocVecEnv(make_env_fns)
    train_env = VecMonitor(train_env)
    # VecNormalize is crucial for many MyoSuite environments
    vec_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval Env
    make_eval_env_fns = [
        make_env(env_id=args.env_id, seed=seed + 1000 + idx, log_dir=log_dir, difficulty=args.difficulty)
        for idx in range(2)
    ]
    eval_env = SubprocVecEnv(make_eval_env_fns)
    eval_env = VecMonitor(eval_env)
    eval_vec_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_vec_env.obs_rms = vec_env.obs_rms

    # Metrics Env (single instance for get_metrics)
    metrics_env = gym.make(args.env_id, **tabletennis_curriculum_kwargs(args.difficulty)).unwrapped

    # 4. Model Initialization
    model = None
    remaining_timesteps = args.total_timesteps

    if args.resume_from_checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.resume_from_checkpoint)
        print(f"Resuming PPO from checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=vec_env)
        
        # Load VecNormalize stats
        vecnorm_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, train_env)
            vec_env.training = True
            model.set_env(vec_env)
            eval_vec_env.obs_rms = vec_env.obs_rms

        current_timesteps = model.num_timesteps
        remaining_timesteps = args.total_timesteps - current_timesteps
        print(f"Resuming training for {remaining_timesteps} steps (target: {args.total_timesteps})")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            use_sde=True,
            sde_sample_freq=4,
            n_steps=args.n_steps,   
            batch_size=args.batch_size,
            tensorboard_log=log_dir,
        )

    # 5. Callbacks
    checkpoint_save_freq = max(1, args.checkpoint_freq // args.num_envs)
    callbacks = [
        PeriodicVideoRecorder(
            video_dir=video_dir,
            env_id=args.env_id,
            record_every_steps=args.render_steps,
            rollout_steps=200,
            camera_ids=[1, 2, 3],
            make_env_fn=lambda: gym.make(args.env_id, **tabletennis_curriculum_kwargs(args.difficulty))
        ),
        WandbCallback(),
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=checkpoint_dir,
            name_prefix="pong_ppo",
        ),
        PeriodicEvaluator(
            eval_vec=eval_vec_env,
            eval_freq=args.eval_freq,
            eval_episodes=10,
            metrics_env=metrics_env,
            verbose=1
        ),
        TqdmProgressCallback(total_steps=args.total_timesteps)
    ]

    # 6. Training
    print(f"Starting training on {args.env_id} for {remaining_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.resume_from_checkpoint)
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 7. Save Model and Normalization Statistics
        model_path = os.path.join(log_dir, "pong_ppo_model")
        model.save(model_path)
        vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        print(f"Training completed. Model saved to {model_path}")

        # 8. Cleanup
        vec_env.close()
        eval_vec_env.close()
        metrics_env.close()
        wandb.finish()

if __name__ == "__main__":
    main()
