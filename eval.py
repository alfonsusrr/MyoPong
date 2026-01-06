import warnings

# supress all warnings
warnings.filterwarnings("ignore")

import argparse
import os
import time
import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from myosuite.utils import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Internal imports
from SAR.SynergyWrapper import SynNoSynWrapper
from modules.models.hierarchical import HierarchicalTableTennisWrapper
from modules.models.lattice import LatticeActorCriticPolicy
from modules.envs.curriculum import tabletennis_curriculum_kwargs
from modules.callback.evaluator import evaluate_policy
from modules.callback.checkpoint import resolve_checkpoint_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Evaluator for Table Tennis")
    
    # Core Evaluation Arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved SB3 model")
    parser.add_argument("--env-id", type=str, default="myoChallengeTableTennisP1-v0", help="Gymnasium environment ID")
    parser.add_argument("--difficulty", type=int, default=1, help="Curriculum difficulty level (0-4)")
    parser.add_argument("--reward-type", type=str, default="standard", help="Reward type (small/standard)")
    parser.add_argument("--eval-envs", type=int, default=12, help="Number of parallel eval envs")
    parser.add_argument("--eval-episodes", type=int, default=400, help="Total eval episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Feature Toggles (must match how the model was trained)
    parser.add_argument("--use-sarl", action="store_true", help="Use Synergy Action Reformulation (SARL)")
    parser.add_argument("--use-hierarchical", action="store_true", help="Use Hierarchical wrapper")
    parser.add_argument("--use-lattice", action="store_true", help="Use LatticeActorCriticPolicy")
    parser.add_argument("--use-lstm", action="store_true", help="Use RecurrentPPO (LSTM)")
    
    # SARL Specific
    parser.add_argument("--sar-dir", type=str, default="SAR/data", help="Directory containing SAR artifacts")
    parser.add_argument("--phi", type=float, default=0.8, help="Synergy blending parameter (SARL)")
    
    # Hierarchical Specific
    parser.add_argument("--update-freq", type=int, default=10, help="Goal update frequency in Hierarchical wrapper")
    
    # Normalization Arguments
    parser.add_argument("--norm-obs", action="store_true", default=True, help="Normalize observations (usually True)")
    parser.add_argument("--clip-obs", type=float, default=10.0, help="Clipping value for observations")
    
    # Rendering
    parser.add_argument("--render", action="store_true", help="Render evaluation (note: might be slow with many envs)")
    parser.add_argument("--video-dir", type=str, default="eval_videos", help="Directory to save evaluation videos")

    return parser.parse_args()

def get_env_kwargs(args: argparse.Namespace):
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

def load_sar_artifacts(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"Loading SAR artifacts from {args.sar_dir}...")
    artifacts = {
        "ica": joblib.load(os.path.join(args.sar_dir, "ica.pkl")),
        "pca": joblib.load(os.path.join(args.sar_dir, "pca.pkl")),
        "scaler": joblib.load(os.path.join(args.sar_dir, "scaler.pkl")),
        "phi": args.phi
    }
    return artifacts

def make_env(args: argparse.Namespace, seed: int, sar_artifacts: Optional[Dict] = None):
    def _init():
        kwargs, alignment_weights = get_env_kwargs(args)
        env = gym.make(args.env_id, **kwargs)
        env.seed(seed)
        
        if args.use_hierarchical:
            env = HierarchicalTableTennisWrapper(env, update_freq=args.update_freq, alignment_weights=alignment_weights)
        
        if args.use_sarl and sar_artifacts:
            env = SynNoSynWrapper(
                env, 
                sar_artifacts["ica"], 
                sar_artifacts["pca"], 
                sar_artifacts["scaler"], 
                sar_artifacts["phi"]
            )
        
        return Monitor(env)
    return _init

def main():
    args = parse_args()

    # Validation logic
    if args.use_lstm and args.use_lattice:
        raise ValueError("LatticeActorCriticPolicy is not compatible with RecurrentPPO (LSTM).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sar_artifacts = load_sar_artifacts(args) if args.use_sarl else None

    # Setup Evaluation Environments
    make_eval_fns = [make_env(args, args.seed + i, sar_artifacts) for i in range(args.eval_envs)]
    eval_env = SubprocVecEnv(make_eval_fns)
    eval_env = VecMonitor(eval_env)
    
    # Resolve checkpoint path (handles directories and files)
    resolved_model_path = resolve_checkpoint_path(args.model_path)
    print(f"Resolved model path: {resolved_model_path}")

    # Handle VecNormalize
    # Search for vecnormalize.pkl in the same directory as the model or its parent
    model_path = Path(resolved_model_path)
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    if not vecnorm_path.exists():
        vecnorm_path = model_path.parent.parent / "vecnormalize.pkl"
    
    if args.norm_obs:
        if vecnorm_path.exists():
            print(f"Loading VecNormalize statistics from {vecnorm_path}")
            eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            print(f"Warning: --norm-obs is True but {vecnorm_path} not found. Evaluation might be inaccurate.")

    # Setup Metrics Env (for get_metrics)
    kwargs, alignment_weights = get_env_kwargs(args)
    metrics_env = gym.make(args.env_id, **kwargs)
    if args.use_hierarchical:
        metrics_env = HierarchicalTableTennisWrapper(metrics_env, update_freq=args.update_freq, alignment_weights=alignment_weights)
    metrics_env = metrics_env.unwrapped

    # Load Model
    print(f"Loading model from {resolved_model_path}...")
    algo_class = RecurrentPPO if args.use_lstm else PPO
    
    # Custom objects mapping to help SB3 find our custom Lattice policy during deserialization
    custom_objects = {}
    if args.use_lattice:
        custom_objects["policy_class"] = LatticeActorCriticPolicy

    model = algo_class.load(resolved_model_path, env=eval_env, device=device, custom_objects=custom_objects)

    # Run Evaluation
    print(f"Starting evaluation: {args.eval_episodes} episodes across {args.eval_envs} environments...")
    t0 = time.time()
    metrics, ep_metrics = evaluate_policy(
        model=model,
        eval_vec=eval_env,
        episodes=args.eval_episodes,
        metrics_env=metrics_env
    )
    duration = time.time() - t0

    # Print Results
    print("\n" + "="*40)
    print(f"{'EVALUATION RESULTS':^40}")
    print("="*40)
    print(f"Model: {resolved_model_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes: {len(ep_metrics)}")
    print("-" * 40)
    
    if metrics:
        # Calculate standard deviation for each metric
        for k, v in metrics.items():
            ep_values = [m[k] for m in ep_metrics if k in m]
            if len(ep_values) > 0:
                std = np.std(ep_values, ddof=1) / np.sqrt(len(ep_values))
                print(f"{k:<15}: {v:>8.4f} +/- {std:>8.4f}")
            else:
                print(f"{k:<15}: {v:>8.4f}")
    else:
        print("No metrics collected.")
    print("="*40)

    # Cleanup
    eval_env.close()
    metrics_env.close()

if __name__ == "__main__":
    main()

