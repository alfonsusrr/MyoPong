#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean PPO training pipeline for MyoSuite using Stable-Baselines3 with Weights & Biases logging
and periodic video rendering. Uses an ActionSpaceWrapper to reduce control dimensionality.

Run:
  python train_myosuite.py \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 100000 \
    --checkpoint-dir /workspace/checkpoints \
    --log-dir /workspace/logs \
    --video-dir /workspace/videos \
    --wandb-project myosuite-ppo \
    --wandb-entity your_wandb_entity

Notes:
- Designed for clarity and debuggability: modular functions, explicit configs, minimal globals.
- Logs at high frequency via SB3 logger + W&B sync; evaluation/rollout video every N steps.
"""

import os
import argparse
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import uuid

import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

# Ensure headless offscreen rendering context for MuJoCo (fallbacks can be set by user)
os.environ.setdefault("MUJOCO_GL", "egl")

from myosuite.utils import gym
from modules.callback.wandb import WandbCallback
from modules.callback.renderer import PeriodicVideoRecorder


class ActionSpaceWrapper(gym.ActionWrapper):
    """Reduce action space dimensionality via muscle synergies + direct mappings.

    This wrapper groups torso muscles into synergy groups and directly maps
    remaining indices, yielding a smaller action vector that's easier to learn.

    Important: The grouping below mirrors the example in MyoSuite.py but can be
    task-specific. Adjust if using a different environment/model.
    """

    def __init__(self, env):
        super().__init__(env)
        # 24 synergy groups + 65 direct entries = 89 reduced controls
        self.syn_action_shape = 24 + 65
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.syn_action_shape,), dtype=np.float32)

        # Group mappings (first 210 torso muscles grouped into 24 synergies)
        self.action_mapping: Dict[int, List[int]] = {
            0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # psoas major right
            1:  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],    # psoas major left
            2:  [22],                                            # RA right
            3:  [23],                                            # RA left
            4:  [24, 25, 26, 27],                                # ILpL right
            5:  [28, 29, 30, 31],                                # ILpL left
            6:  [32, 33, 34, 35, 36, 37, 38, 39],                # ILpT right
            7:  [40, 41, 42, 43, 44, 45, 46, 47],                # ILpT left
            8:  list(range(48, 69)),                             # LTpT right
            9:  list(range(69, 90)),                             # LTpT left
            10: [90, 91, 92, 93, 94],                            # LTpL right
            11: [95, 96, 97, 98, 99],                            # LTpL left
            12: [100, 101, 102, 103, 104, 105, 106],             # QL_post right
            13: [107, 108, 109, 110, 111, 112, 113],             # QL_post left
            14: [114, 115, 116, 117, 118],                       # QL_mid right
            15: [119, 120, 121, 122, 123],                       # QL_mid left
            16: [124, 125, 126, 127, 128, 129],                  # QL_ant right
            17: [130, 131, 132, 133, 134, 135],                  # QL_ant left
            18: list(range(136, 161)),                           # MF right
            19: list(range(161, 186)),                           # MF left
            20: [186, 187, 188, 189, 190, 191],                  # EO right
            21: [192, 193, 194, 195, 196, 197],                  # IO right (note: label per source)
            22: [198, 199, 200, 201, 202, 203],                  # EO left
            23: [204, 205, 206, 207, 208, 209],                  # IO left
        }

        # Direct mapping for next 65 controls; align to env's action space shape.
        # This mirrors the example; if using a different env, verify indices.
        for i in range(24, 89):
            # Following MyoSuite.py example comment, but ensure bounds within action space
            idx = i + 169
            if idx < self.env.action_space.shape[0]:
                self.action_mapping[i] = [idx]

    def action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape[0] == self.syn_action_shape, (
            f"Expected action of shape {(self.syn_action_shape,)}, got {action.shape}")
        full_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for reduced_index, indices in self.action_mapping.items():
            # Broadcast single scalar to target indices
            full_action[indices] = action[reduced_index]
        return full_action

    def apply_curriculum(self, callable_name: str, value: float) -> None:
        """Proxy to call a curriculum setter on the base env if available.

        This lets curriculum adjustors call into the underlying MyoSuite env
        even though it is wrapped by this `ActionSpaceWrapper`.
        """
        base = self.unwrapped
        # Specialized auto strategy for TableTennis
        if callable_name == 'tabletennis_auto':
            try:
                self.apply_tabletennis_curriculum(float(value))
                return
            except Exception:
                pass
        if hasattr(base, callable_name):
            try:
                getattr(base, callable_name)(value)
            except TypeError:
                # Fallback: some setters might accept (value,)
                getattr(base, callable_name)(value)
        else:
            # Best-effort: try common attribute names
            for attr in [callable_name, 'difficulty', 'level']:
                if hasattr(base, attr):
                    try:
                        setattr(base, attr, value)
                        break
                    except Exception:
                        pass

    def apply_tabletennis_curriculum(self, level: float) -> None:
        """Environment-aware difficulty shaping for TableTennisEnvV0.

        level in [0,1]. We increase stochasticity and physical variability:
        - ramp rally_count 1->5
        - increase qpos_noise_range (0 -> 0.05 fraction of joint ranges)
        - add serve position variation via ball_xyz_range around current pos
        - enable variable ball_qvel after mid levels
        - vary ball friction and paddle mass slightly
        """
        base = self.unwrapped
        # Guard: only operate if expected fields exist
        if not hasattr(base, 'sim') or not hasattr(base, 'id_info'):
            return

        lvl = float(np.clip(level, 0.0, 1.0))

        # Rally count: integer 1..5
        target_rallies = int(round(1 + lvl * 4))
        try:
            base.rally_count = max(1, target_rallies)
        except Exception:
            pass

        # Joint init noise fraction: 0..0.05
        noise = 0.05 * lvl
        try:
            base.qpos_noise_range = {'low': -noise, 'high': noise}
        except Exception:
            pass

        # Serve position variation box around current ball position
        try:
            pos = base.sim.model.body_pos[base.id_info.ball_bid].copy()
            delta = np.array([0.05, 0.20, 0.02], dtype=float) * lvl  # x,y,z spread
            base.ball_xyz_range = {
                'low': (pos - delta).tolist(),
                'high': (pos + delta).tolist(),
            }
        except Exception:
            pass

        # Velocity variability toggled on after 0.3 level
        try:
            base.ball_qvel = bool(lvl >= 0.3)
        except Exception:
            pass

        # Ball friction slight variability (scale +/-10% at max level)
        try:
            fric = base.sim.model.geom_friction[base.id_info.ball_gid].copy()
            scale = 0.1 * lvl
            low = (fric * (1.0 - scale)).tolist()
            high = (fric * (1.0 + scale)).tolist()
            base.ball_friction_range = {'low': low, 'high': high}
        except Exception:
            pass

        # Paddle mass slight variability (+/- 10% at max level)
        try:
            mass = float(base.sim.model.body_mass[base.id_info.paddle_bid])
            delta_mass = mass * 0.1 * lvl
            base.paddle_mass_range = [mass - delta_mass, mass + delta_mass]
        except Exception:
            pass


def evaluate_policy(model: PPO, eval_vec: VecMonitor, episodes: int) -> Dict[str, float]:
    """Run evaluation on an existing vectorized env and return averaged metrics.

    Accumulates step-wise reward dictionaries from infos into per-episode `paths`:
      path = { 'env_infos': { 'rwd_dict': { key: np.array([...]) } } }

    Returns a dict of averaged metrics across all completed episodes. The provided
    `eval_vec` is reset at the start and reused across calls for performance.
    """
    # Per-env accumulators of rwd_dict over time
    rwd_accum: List[List[Dict[str, Any]]] = [[] for _ in range(eval_vec.num_envs)]
    collected = 0
    all_paths: List[Dict[str, Any]] = []

    obs = eval_vec.reset()
    while collected < episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec.step(actions)

        # Accumulate step rwd_dicts
        for i in range(eval_vec.num_envs):
            info_i = infos[i] if isinstance(infos, (list, tuple)) else {}
            rwd_dict = info_i.get('rwd_dict') if isinstance(info_i, dict) else None
            if isinstance(rwd_dict, dict):
                rwd_accum[i].append(rwd_dict)

        # Handle episode terminations
        for i, done in enumerate(dones):
            if done:
                # Build path with stacked rwd_dict arrays
                steps_dicts = rwd_accum[i]
                stacked: Dict[str, np.ndarray] = {}
                if len(steps_dicts) > 0:
                    keys = set().union(*[d.keys() for d in steps_dicts])
                    for k in keys:
                        vals = [d.get(k) for d in steps_dicts]
                        numeric_vals: List[float] = []
                        for v in vals:
                            if isinstance(v, (int, float, np.floating)):
                                numeric_vals.append(float(v))
                            elif isinstance(v, np.ndarray) and v.size == 1:
                                numeric_vals.append(float(v.item()))
                        if len(numeric_vals) > 0:
                            stacked[k] = np.array(numeric_vals, dtype=np.float32)
                path = {'env_infos': {'rwd_dict': stacked}}
                all_paths.append(path)

                # Reset accumulator for this env and increment count
                rwd_accum[i] = []
                collected += 1
                if collected >= episodes:
                    break

    avg_metrics: Dict[str, float] = {}
    # Defer to a provided metrics_env via closure/global; PeriodicEvaluator will pass it in
    metrics_env = getattr(evaluate_policy, "_metrics_env", None)
    try:
        if len(all_paths) > 0 and metrics_env is not None and hasattr(metrics_env, "get_metrics"):
            avg = metrics_env.get_metrics(all_paths)
            if isinstance(avg, dict):
                avg_metrics = {k: float(v) for k, v in avg.items() if isinstance(v, (int, float, np.floating))}
    except Exception:
        pass
        
    return avg_metrics


class PeriodicEvaluator(BaseCallback):
    """Periodically evaluate the current policy and log averaged metrics."""

    def __init__(self, env_id: str, eval_freq: int = 10000, eval_envs: int = 2, eval_episodes: int = 10, seed: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.env_id = env_id
        self.eval_freq = eval_freq
        self.eval_envs = eval_envs
        self.eval_episodes = eval_episodes
        self.seed = seed
        self._last_eval_step = 0
        # Build a persistent eval vec env once to avoid repeated construction overhead
        # Use multiprocessing when more than one eval env is requested
        self._eval_vec = build_vec_env(self.env_id, self.eval_envs, self.seed + 12345)
        # Build a single base env for metrics computation because SubprocVecEnv workers
        # cannot expose methods like get_metrics directly
        base_env = gym.make(self.env_id)
        self._metrics_env = base_env.unwrapped
        # Attach metrics env for evaluate_policy to use
        setattr(evaluate_policy, "_metrics_env", self._metrics_env)

    def _on_step(self) -> bool:
        # Align last eval step on resume to prevent immediate long evaluations.
        if self._last_eval_step == 0 and self.num_timesteps > 0:
            self._last_eval_step = self.num_timesteps
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            try:
                t0 = time.time()
                metrics = evaluate_policy(
                    model=self.model,
                    eval_vec=self._eval_vec,
                    episodes=self.eval_episodes,
                )
                duration_s = time.time() - t0
                if metrics:
                    # Log to W&B with eval/ prefix
                    log_data = {f"eval/{k}": v for k, v in metrics.items()}
                    log_data["eval/duration_s"] = duration_s
                    wandb.log(log_data, step=self.num_timesteps)
                    if self.verbose:
                        print(f"[Eval] step={self.num_timesteps} metrics={metrics} duration_s={duration_s:.2f}")
            except Exception as e:
                if self.verbose:
                    print(f"[Eval] Evaluation failed: {e}")
        return True

    def __del__(self):
        try:
            if hasattr(self, "_eval_vec") and self._eval_vec is not None:
                self._eval_vec.close()
            if hasattr(self, "_metrics_env") and self._metrics_env is not None:
                try:
                    self._metrics_env.close()
                except Exception:
                    pass
        except Exception:
            pass


def make_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = ActionSpaceWrapper(env)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(env_id: str, num_envs: int, seed: int) -> VecMonitor:
    env_fns = [make_env(env_id, seed + i) for i in range(num_envs)]
    # Default to SubprocVecEnv when we have more than one environment to leverage multiple CPU cores
    use_subproc = os.environ.get("MYOSUITE_USE_SUBPROC", "1") == "1"
    print(f"Using SubprocVecEnv: {use_subproc} and num_envs: {num_envs}")
    if use_subproc and num_envs > 1:
        vec = SubprocVecEnv(env_fns, start_method="fork")
    else:
        vec = DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    return vec


class CurriculumManager(BaseCallback):
    """Periodically evaluate and promote task difficulty via an env setter.

    Promotion occurs when a chosen metric (from env.get_metrics) exceeds a
    threshold for N consecutive evaluations. Difficulty values are linearly
    interpolated from start->end across a fixed number of steps (levels).
    """

    def __init__(
        self,
        training_vec_env: VecMonitor,
        env_id: str,
        callable_name: str,
        start_value: float,
        end_value: float,
        steps: int,
        metric_key: str,
        threshold: float,
        patience: int,
        eval_freq: int,
        eval_envs: int,
        eval_episodes: int,
        seed: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.training_vec_env = training_vec_env
        self.env_id = env_id
        self.callable_name = callable_name
        self.start_value = start_value
        self.end_value = end_value
        self.steps = max(1, steps)
        self.metric_key = metric_key
        self.threshold = threshold
        self.patience = max(1, patience)
        self.eval_freq = eval_freq
        self.eval_envs = eval_envs
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.level = 0
        self._passes = 0
        self._last_eval_step = 0

        # Persistent eval env and metrics env to compute metrics
        self._eval_vec = build_vec_env(self.env_id, self.eval_envs, self.seed + 54321)
        base_env = gym.make(self.env_id)
        self._metrics_env = base_env.unwrapped
        setattr(evaluate_policy, "_metrics_env", self._metrics_env)

        # Apply initial difficulty
        self._apply_current_level()

    def _on_step(self) -> bool:
        if self._last_eval_step == 0 and self.num_timesteps > 0:
            self._last_eval_step = self.num_timesteps
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            try:
                t0 = time.time()
                metrics = evaluate_policy(
                    model=self.model,
                    eval_vec=self._eval_vec,
                    episodes=self.eval_episodes,
                )
                duration_s = time.time() - t0
                metric_val = float(metrics.get(self.metric_key, 0.0)) if isinstance(metrics, dict) else 0.0
                wandb.log({
                    "curriculum/metric": metric_val,
                    "curriculum/level": self.level,
                    "curriculum/eval_duration_s": duration_s,
                }, step=self.num_timesteps)

                if metric_val >= self.threshold:
                    self._passes += 1
                else:
                    self._passes = 0

                if self._passes >= self.patience and self.level < (self.steps - 1):
                    self.level += 1
                    self._passes = 0
                    self._apply_current_level()
                    if self.verbose:
                        print(f"[Curriculum] Promoted to level {self.level} at step {self.num_timesteps}")
            except Exception as e:
                if self.verbose:
                    print(f"[Curriculum] Evaluation failed: {e}")
        return True

    def _apply_current_level(self) -> None:
        frac = 0.0 if self.steps <= 1 else float(self.level) / float(self.steps - 1)
        value = (1.0 - frac) * self.start_value + frac * self.end_value
        try:
            # Apply to all training envs via wrapper helper
            self.training_vec_env.env_method('apply_curriculum', self.callable_name, float(value), indices=None)
            wandb.log({"curriculum/value": float(value)}, step=self.num_timesteps)
        except Exception as e:
            if self.verbose:
                print(f"[Curriculum] Failed to apply value {value}: {e}")

    def __del__(self):
        try:
            if hasattr(self, "_eval_vec") and self._eval_vec is not None:
                self._eval_vec.close()
            if hasattr(self, "_metrics_env") and self._metrics_env is not None:
                try:
                    self._metrics_env.close()
                except Exception:
                    pass
        except Exception:
            pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MyoSuite with reduced action space and W&B logging")
    parser.add_argument('--env-id', type=str, default='myoChallengeTableTennisP1-v0', help='Gymnasium environment ID')
    parser.add_argument('--total-timesteps', type=int, required=True, help='Total training timesteps')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, required=True, help='Directory for SB3 logs')
    parser.add_argument('--video-dir', type=str, required=True, help='Directory to save training videos')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel envs')
    parser.add_argument('--no-subproc', action='store_true', help='Disable subprocess vectorized envs even if num-envs>1')
    parser.add_argument('--checkpoint-freq', type=int, default=50000, help='Save checkpoint every N steps')
    parser.add_argument('--video-freq', type=int, default=50000, help='Record rollout video every N steps')
    parser.add_argument('--video-rollout-steps', type=int, default=600, help='Steps to render in each video')
    parser.add_argument('--wandb-project', type=str, required=True, help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity (team/user)')
    parser.add_argument('--wandb-name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluate policy every N steps')
    parser.add_argument('--eval-envs', type=int, default=2, help='Number of parallel eval envs')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Total eval episodes per evaluation run')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    # Curriculum learning configuration (optional)
    parser.add_argument('--curriculum-enable', action='store_true', help='Enable curriculum learning during training')
    parser.add_argument('--curriculum-callable', type=str, default=None, help='Env method to set difficulty (e.g., set_difficulty) or use tabletennis_auto')
    parser.add_argument('--curriculum-start', type=float, default=0.1, help='Starting difficulty value passed to curriculum callable')
    parser.add_argument('--curriculum-end', type=float, default=1.0, help='Final difficulty value passed to curriculum callable')
    parser.add_argument('--curriculum-steps', type=int, default=5, help='Number of curriculum promotion steps (levels)')
    parser.add_argument('--curriculum-metric', type=str, default='score', help='Evaluation metric key to gate promotions (from env.get_metrics)')
    parser.add_argument('--curriculum-threshold', type=float, default=0.7, help='Threshold on metric to promote to next level')
    parser.add_argument('--curriculum-patience', type=int, default=2, help='Consecutive passes required before promotion')
    parser.add_argument('--curriculum-eval-freq', type=int, default=20000, help='How often (steps) to run curriculum evaluation')
    parser.add_argument('--curriculum-eval-envs', type=int, default=2, help='Number of envs for curriculum evaluation')
    parser.add_argument('--curriculum-eval-episodes', type=int, default=10, help='Episodes per curriculum evaluation')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Create unique run directory to avoid overwriting previous runs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    run_id = f"run-{timestamp}-{short_id}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
    log_dir = os.path.join(args.log_dir, run_id)
    video_dir = os.path.join(args.video_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Init W&B
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))

    # Build environment
    vec_env = build_vec_env(args.env_id, args.num_envs if not args.no_subproc else 1, args.seed)

    # SB3 logger configuration (stdout + tensorboard-like files in log_dir)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # PPO model configuration - tune as needed
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found at: {args.resume_from_checkpoint}")
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        model = PPO.load(
            args.resume_from_checkpoint,
            env=vec_env,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            batch_size=args.num_envs * 2048,
            n_steps=2048,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            device="cpu"
        )
    model.set_logger(new_logger)

    # Callbacks: checkpointing, W&B logging, periodic video
    callbacks: List[BaseCallback] = []

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // max(1, args.num_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo_myo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    callbacks.append(WandbCallback(verbose=0))
    callbacks.append(PeriodicVideoRecorder(
        video_dir,
        env_id=args.env_id,
        record_every_steps=args.video_freq,
        rollout_steps=args.video_rollout_steps,
        wrap_env_fn=lambda env: ActionSpaceWrapper(env),
        verbose=1
    ))
    callbacks.append(PeriodicEvaluator(env_id=args.env_id, eval_freq=args.eval_freq, eval_envs=args.eval_envs, eval_episodes=args.eval_episodes, seed=args.seed, verbose=1))

    # Optional: Curriculum learning
    if args.curriculum_enable and args.curriculum_callable:
        curriculum_cb = CurriculumManager(
            training_vec_env=vec_env,
            env_id=args.env_id,
            callable_name=args.curriculum_callable,
            start_value=args.curriculum_start,
            end_value=args.curriculum_end,
            steps=args.curriculum_steps,
            metric_key=args.curriculum_metric,
            threshold=args.curriculum_threshold,
            patience=args.curriculum_patience,
            eval_freq=args.curriculum_eval_freq,
            eval_envs=args.curriculum_eval_envs,
            eval_episodes=args.curriculum_eval_episodes,
            seed=args.seed,
            verbose=1,
        )
        callbacks.append(curriculum_cb)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=not args.resume_from_checkpoint)

    # Final save
    final_path = os.path.join(checkpoint_dir, 'ppo_myo_final')
    model.save(final_path)
    wandb.summary['final_model_path'] = final_path
    wandb.summary['run_id'] = run_id

    vec_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
