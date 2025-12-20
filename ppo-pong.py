import os
import time
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb

from modules.callback.renderer import PeriodicVideoRecorder
from modules.callback.wandb import WandbCallback
from modules.envs.curriculum import tabletennis_curriculum_kwargs

def main():
    # 1. Configuration
    env_id = "myoChallengePongP0-v0"
    total_timesteps = 1_000_000
    seed = 42
    run_id = f"pong-ppo-{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", "pong_ppo_simple", run_id)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Wandb Initialization
    wandb.init(
        project="myosuite-ghost-pong",
        name=run_id,
        config={
            "env_id": env_id,
            "total_timesteps": total_timesteps,
            "seed": seed,
            "num_envs": 8,
        },
    )

    # 3. Environment Setup
    # We use a single environment for this simple example
    def make_env():
        # Get curriculum kwargs for ball randomization (ball_xyz_range, ball_qvel)
        kwargs = tabletennis_curriculum_kwargs(difficulty=3)
        env = gym.make(env_id, **kwargs)
        return env

    env = DummyVecEnv([make_env for _ in range(8)])

    env = VecMonitor(env)
    # VecNormalize is crucial for many MyoSuite environments due to observation/reward scales
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 4. Model Initialization
    # Using hyperparameters often effective for MyoSuite (e.g., gSDE)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        use_sde=True,              # Smooth exploration for muscle control
        sde_sample_freq=4,
        n_steps=2048,              # Collected steps per update
        batch_size=256,            # Mini-batch size
        tensorboard_log=log_dir,
    )

    # 5. Callbacks
    video_recorder = PeriodicVideoRecorder(
        video_dir=os.path.join(log_dir, "videos"),
        env_id=env_id,
        record_every_steps=500000,
        rollout_steps=200,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 250000 // 8),
        save_path=checkpoint_dir,
        name_prefix="pong_ppo",
    )

    callbacks = [video_recorder, WandbCallback(), checkpoint_callback]

    # 6. Training
    print(f"Starting training on {env_id} for {total_timesteps} steps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 7. Save Model and Normalization Statistics
        model_path = os.path.join(log_dir, "pong_ppo_model")
        model.save(model_path)
        env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        print(f"Training completed. Model saved to {model_path}")

        # 8. Cleanup
        env.close()
        wandb.finish()

if __name__ == "__main__":
    main()

