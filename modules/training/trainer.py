import os
import time
import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from myosuite.utils import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Internal imports
from SAR.SynergyWrapper import SynNoSynWrapper
from modules.models.hierarchical import HierarchicalTableTennisWrapper
from modules.models.lattice import LatticeActorCriticPolicy
from modules.callback.progress import TqdmProgressCallback
from modules.callback.evaluator import PeriodicEvaluator
from modules.callback.wandb import WandbCallback
from modules.callback.renderer import PeriodicVideoRecorder
from modules.callback.checkpoint import SaveVecNormalizeCallback, ScoreThresholdSaveCallback, resolve_checkpoint_path
from modules.envs.curriculum import tabletennis_curriculum_kwargs

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class TableTennisTrainer:
    def __init__(self, args: Any):
        self.args = args
        self.run_id = self._generate_run_id()
        self.log_dir = os.path.abspath(os.path.join(args.log_dir, self.run_id))
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.video_dir = os.path.join(self.log_dir, "videos")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.device = self._get_device()
        self.sar_artifacts = self._load_sar_artifacts() if args.use_sarl else None
        
        self.vec_env = None
        self.eval_vec_env = None
        self.metrics_env = None
        self.wandb_module = None

    def _generate_run_id(self) -> str:
        prefix = "run-ppo"
        if self.args.use_hierarchical: prefix += "-h"
        if self.args.use_sarl: prefix += "-sarl"
        if self.args.use_lattice: prefix += "-lattice"
        if self.args.use_lstm: prefix += "-lstm"
        
        return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-lvl{self.args.difficulty}"

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        print("Using CPU device.")
        return torch.device("cpu")

    def _load_sar_artifacts(self) -> Dict[str, Any]:
        print(f"Loading SAR artifacts from {self.args.sar_dir}...")
        artifacts = {
            "ica": joblib.load(os.path.join(self.args.sar_dir, "ica.pkl")),
            "pca": joblib.load(os.path.join(self.args.sar_dir, "pca.pkl")),
            "scaler": joblib.load(os.path.join(self.args.sar_dir, "scaler.pkl")),
            "phi": 0.8
        }
        print("SAR artifacts loaded.")
        return artifacts

    def prepare_env(self, env_id: str, difficulty: int) -> Any:
        kwargs = tabletennis_curriculum_kwargs(difficulty)
        try:
            env = gym.make(env_id, **kwargs)
        except TypeError:
            env = gym.make(env_id)
        return env

    def make_env(self, seed: int, is_eval: bool = False) -> Callable[[], Monitor]:
        def _init():
            env = self.prepare_env(self.args.env_id, self.args.difficulty)
            env.seed(seed)
            
            if self.args.use_hierarchical:
                env = HierarchicalTableTennisWrapper(env, update_freq=self.args.update_freq)
            
            if self.args.use_sarl and self.sar_artifacts:
                env = SynNoSynWrapper(
                    env, 
                    self.sar_artifacts["ica"], 
                    self.sar_artifacts["pca"], 
                    self.sar_artifacts["scaler"], 
                    self.sar_artifacts["phi"]
                )
            
            suffix = "eval" if is_eval else "train"
            return Monitor(env, filename=os.path.join(self.log_dir, f"monitor_{seed}_{suffix}.csv"))
        return _init

    def setup_envs(self):
        # Training Envs
        make_env_fns = [self.make_env(self.args.seed + i) for i in range(self.args.num_envs)]
        train_env = SubprocVecEnv(make_env_fns)
        train_env = VecMonitor(train_env)
        self.vec_env = VecNormalize(
            train_env,
            training=True,
            norm_obs=self.args.norm_obs,
            norm_reward=self.args.norm_reward,
            clip_obs=self.args.clip_obs,
            clip_reward=self.args.clip_reward
        )

        # Eval Envs
        make_eval_fns = [self.make_env(self.args.seed + 1000 + i, is_eval=True) for i in range(self.args.eval_envs)]
        eval_env = SubprocVecEnv(make_eval_fns)
        eval_env = VecMonitor(eval_env)
        self.eval_vec_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=self.args.norm_obs,
            norm_reward=False,
            clip_obs=self.args.clip_obs
        )
        self.eval_vec_env.obs_rms = self.vec_env.obs_rms

        # Metrics Env
        self.metrics_env = self.prepare_env(self.args.env_id, self.args.difficulty)
        if self.args.use_hierarchical:
            self.metrics_env = HierarchicalTableTennisWrapper(self.metrics_env, update_freq=self.args.update_freq)
        self.metrics_env = self.metrics_env.unwrapped

    def _get_activation_fn(self) -> Any:
        if self.args.activation_fn is None:
            # Default logic based on setup
            if self.args.use_lattice or self.args.use_hierarchical:
                return torch.nn.SiLU
            return torch.nn.Tanh
        
        mapping = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU,
            "elu": torch.nn.ELU
        }
        return mapping[self.args.activation_fn.lower()]

    def setup_model(self):
        policy_kwargs = {}
        activation_fn = self._get_activation_fn()
        log_std_init = self.args.log_std_init
        
        # Determine Algorithm and Policy
        if self.args.use_lstm:
            algo_class = RecurrentPPO
            policy_type = "MlpLstmPolicy"
            policy_kwargs.update({
                "lstm_hidden_size": self.args.lstm_hidden_size,
                "n_lstm_layers": self.args.n_lstm_layers,
                "activation_fn": activation_fn,
            })
            if log_std_init is not None:
                policy_kwargs["log_std_init"] = log_std_init
        else:
            algo_class = PPO
            if self.args.use_lattice:
                policy_type = LatticeActorCriticPolicy
                policy_kwargs.update({
                    "use_lattice": True,
                    "net_arch": dict(pi=[512, 512], vf=[256, 256]),
                    "activation_fn": activation_fn,
                    "ortho_init": True,
                    "std_clip": (self.args.std_clip_min, self.args.std_clip_max),
                    "std_reg": self.args.std_reg,
                    "alpha": self.args.lattice_alpha,
                })
                if log_std_init is not None:
                    policy_kwargs["log_std_init"] = log_std_init
            elif self.args.use_hierarchical:
                policy_type = self.args.policy
                policy_kwargs.update({
                    "net_arch": dict(pi=[512, 512], vf=[256, 256]),
                    "activation_fn": activation_fn,
                    "ortho_init": True,
                    "log_std_init": log_std_init if log_std_init is not None else -1.0,
                })
            else:
                policy_type = self.args.policy
                policy_kwargs.update({
                    "log_std_init": log_std_init if log_std_init is not None else -2,
                    "net_arch": [256, 256],
                    "activation_fn": activation_fn,
                })

        # Load or Create Model
        if self.args.resume_from_checkpoint:
            checkpoint_path = resolve_checkpoint_path(self.args.resume_from_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Load VecNormalize if it exists
            vecnorm_path = Path(checkpoint_path).parent.parent / "vecnormalize.pkl"
            if vecnorm_path.exists():
                print(f"Loading VecNormalize statistics from {vecnorm_path}")
                self.vec_env = VecNormalize.load(str(vecnorm_path), self.vec_env.venv)
                self.vec_env.training = True
                self.eval_vec_env.obs_rms = self.vec_env.obs_rms
            
            self.model = algo_class.load(checkpoint_path, env=self.vec_env, device=self.device)
        else:
            self.model = algo_class(
                policy=policy_type,
                env=self.vec_env,
                learning_rate=linear_schedule(self.args.learning_rate) if not self.args.use_lstm else self.args.learning_rate,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                ent_coef=self.args.ent_coef,
                verbose=1,
                seed=self.args.seed,
                tensorboard_log=os.path.abspath(self.args.tensorboard_log) if self.args.tensorboard_log else None,
                device=self.device,
                use_sde=not self.args.use_lstm,
                sde_sample_freq=4 if not self.args.use_lstm else -1,
                policy_kwargs=policy_kwargs,
            )

    def setup_callbacks(self):
        checkpoint_save_freq = max(1, self.args.checkpoint_freq // self.args.num_envs)
        
        self.callbacks = [
            SaveVecNormalizeCallback(
                save_freq=checkpoint_save_freq,
                save_path=self.checkpoint_dir,
                name_prefix="ppo_model",
            ),
            TqdmProgressCallback(total_steps=self.args.total_timesteps)
        ]

        if self.args.wandb_project:
            import wandb
            self.wandb_module = wandb
            self.wandb_module.init(
                project=self.args.wandb_project,
                name=self.run_id,
                config=vars(self.args),
                sync_tensorboard=True if self.args.tensorboard_log else False
            )
            self.callbacks.append(WandbCallback())

        evaluator = PeriodicEvaluator(
            eval_vec=self.eval_vec_env,
            eval_freq=self.args.eval_freq,
            eval_episodes=self.args.eval_episodes,
            metrics_env=self.metrics_env,
            verbose=1
        )
        self.callbacks.append(evaluator)

        if self.args.score_threshold > 0:
            self.callbacks.append(ScoreThresholdSaveCallback(
                evaluator=evaluator,
                save_path=self.checkpoint_dir,
                threshold=self.args.score_threshold,
                name_prefix="ppo_model",
                verbose=1
            ))

        if self.args.render_steps > 0:
            def _wrap_env_for_rendering(env):
                if self.args.use_hierarchical:
                    env = HierarchicalTableTennisWrapper(env, update_freq=self.args.update_freq)
                if self.args.use_sarl and self.sar_artifacts:
                    env = SynNoSynWrapper(
                        env, 
                        self.sar_artifacts["ica"], 
                        self.sar_artifacts["pca"], 
                        self.sar_artifacts["scaler"], 
                        self.sar_artifacts["phi"]
                    )
                return Monitor(env, filename=os.path.join(self.log_dir, "renderer_monitor.csv"))

            self.callbacks.append(PeriodicVideoRecorder(
                video_dir=self.video_dir,
                env_id=self.args.env_id,
                record_every_steps=self.args.render_steps,
                rollout_steps=self.args.rollout_steps,
                wrap_env_fn=_wrap_env_for_rendering,
                verbose=1,
                make_env_fn=lambda: self.prepare_env(self.args.env_id, self.args.difficulty)
            ))

    def train(self):
        remaining_timesteps = self.args.total_timesteps - self.model.num_timesteps
        if remaining_timesteps <= 0:
            print("Target timesteps already reached.")
            return

        try:
            self.model.learn(
                total_timesteps=remaining_timesteps,
                callback=self.callbacks,
                reset_num_timesteps=not bool(self.args.resume_from_checkpoint),
            )
        finally:
            self.cleanup()

    def cleanup(self):
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.save(os.path.join(self.log_dir, "vecnormalize.pkl"))
        
        save_path = self.args.save_path or os.path.join(self.log_dir, "final_model")
        self.model.save(save_path)
        
        try:
            self.vec_env.close()
            self.eval_vec_env.close()
            if self.metrics_env: self.metrics_env.close()
        except:
            pass
            
        print(f"Model saved to: {save_path}")
        if self.wandb_module:
            self.wandb_module.finish()

