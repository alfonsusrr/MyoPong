import os
import argparse
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from myosuite.utils import gym
from tqdm import tqdm

class ParallelCollectWrapper(gym.Wrapper):
    """
    Wrapper to:
    1. Collect muscle activations and put them in info dict.
    2. Convert dict observations to vectors if environment supports it (obsdict2obsvec).
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            obs, info = ret
        else:
            obs = ret
            info = {}
        
        obs = self._process_obs(obs)
        
        if isinstance(ret, tuple):
            return obs, info
        return obs

    def step(self, action):
        ret = self.env.step(action)
        # Handle 4 or 5 tuple
        if len(ret) == 5:
            obs, reward, done, truncated, info = ret
        else:
            obs, reward, done, info = ret
            truncated = False
        
        # Capture activation
        sim = getattr(self.env.unwrapped, "sim", None)
        if sim is not None and hasattr(sim, "data") and sim.data.act is not None:
            info['muscle_activations'] = sim.data.act.copy()
        
        obs = self._process_obs(obs)
        
        # Reconstruct return
        if len(ret) == 5:
            return obs, reward, done, truncated, info
        return obs, reward, done, info

    def _process_obs(self, obs):
        if isinstance(obs, dict) and hasattr(self.env.unwrapped, "obsdict2obsvec"):
             _, obs_vec = self.env.unwrapped.obsdict2obsvec(self.env.unwrapped.obs_dict, self.env.unwrapped.obs_keys)
             return obs_vec
        return obs

def resolve_checkpoint_path(checkpoint_target: str) -> Path:
  target = Path(checkpoint_target).expanduser()
  if target.is_dir():
    checkpoints = sorted(target.glob("*.zip"))
    if not checkpoints:
      raise FileNotFoundError(f"No checkpoint archives found in {target}")
    return checkpoints[-1].resolve()

  if target.is_file():
    return target.resolve()

  if target.suffix != ".zip":
    zipped_candidate = target.with_suffix(".zip")
    if zipped_candidate.is_file():
      return zipped_candidate.resolve()

  raise FileNotFoundError(f"Checkpoint path {checkpoint_target} does not exist")

def make_env(env_id: str):
    def _init():
        try:
            env = gym.make(env_id)
        except TypeError:
            env = gym.make(env_id)
        return ParallelCollectWrapper(env)
    return _init

def collect_activations(checkpoint_path: Path, env_id: str, episodes: int, percentile: int, deterministic: bool = False, n_envs: int = 1) -> np.ndarray:
  print(f"Loading model from {checkpoint_path}...")
  
  # For preview, we can use a single environment or vectorized. 
  # Using vectorized for preview as well to be consistent and fast.
  print(f"Creating {n_envs} environments...")
  vec_env = make_vec_env(make_env(env_id), n_envs=n_envs, vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
  
  # Load model
  # Note: PPO.load with env=vec_env attaches the env to the model.
  model = PPO.load(str(checkpoint_path), env=vec_env)
  
  print(f"Calculating reward threshold from 100 preview episodes...")
  preview_rewards = []
  
  obs = vec_env.reset()
  current_rewards = np.zeros(n_envs)
  
  # Calculate how many episodes we need to finish for preview
  preview_episodes_needed = 100
  preview_episodes_done = 0
  
  pbar = tqdm(total=preview_episodes_needed, desc="Preview")
  
  while preview_episodes_done < preview_episodes_needed:
      action, _ = model.predict(obs, deterministic=deterministic)
      obs, rewards, dones, infos = vec_env.step(action)
      
      current_rewards += rewards
      
      for i in range(n_envs):
          if dones[i]:
              preview_rewards.append(current_rewards[i])
              current_rewards[i] = 0
              preview_episodes_done += 1
              pbar.update(1)
              
  pbar.close()
  # Trim to exactly 100 if we went over
  preview_rewards = preview_rewards[:100]
  
  reward_threshold = np.percentile(preview_rewards, percentile)
  print(f"Reward threshold ({percentile}th percentile): {reward_threshold:.2f}")

  print(f"Collecting data from {episodes} episodes...")
  solved_acts = []
  success_count = 0
  episodes_completed = 0
  
  # Reset buffers
  current_rewards = np.zeros(n_envs)
  current_acts = [[] for _ in range(n_envs)]
  
  # Note: vec_env auto-resets, but we might have partial trajectories from preview.
  # It's safer to reset environments or just continue. 
  # Since obs is continuous from previous loop, we can continue.
  # But current_acts is empty, so we might miss the start of the current episodes.
  # Better to force reset to ensure we capture full episodes.
  obs = vec_env.reset()
  current_rewards = np.zeros(n_envs)
  
  pbar = tqdm(total=episodes, desc="Collection")
  
  while episodes_completed < episodes:
      action, _ = model.predict(obs, deterministic=deterministic)
      obs, rewards, dones, infos = vec_env.step(action)
      
      for i in range(n_envs):
          # Collect activation from info
          # If done[i] is True, info[i] contains the info from the last step (terminal state)
          if 'muscle_activations' in infos[i]:
              current_acts[i].append(infos[i]['muscle_activations'])
          
          current_rewards[i] += rewards[i]
          
          if dones[i]:
              # Check threshold
              if current_rewards[i] > reward_threshold:
                  solved_acts.extend(current_acts[i])
                  success_count += 1
              
              episodes_completed += 1
              pbar.update(1)
              
              # Reset for this env
              current_acts[i] = []
              current_rewards[i] = 0
  
  pbar.close()
  vec_env.close()
  
  print(f"Collected data from {success_count} successful episodes (out of {episodes_completed}).")
  return np.array(solved_acts)

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Collect muscle activations from PPO policy")
  parser.add_argument("--env-id", type=str, default="myoChallengeTableTennisP1-v0", help="Environment ID")
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint")
  parser.add_argument("--output", type=str, default="activations.npy", help="Output path for .npy file")
  parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run")
  parser.add_argument("--percentile", type=int, default=80, help="Percentile for reward threshold")
  parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
  parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
  return parser.parse_args()

def main():
  args = parse_args()
  checkpoint_path = resolve_checkpoint_path(args.checkpoint)
  
  activations = collect_activations(
    checkpoint_path, 
    args.env_id, 
    args.episodes, 
    args.percentile,
    args.deterministic,
    args.num_envs
  )
  
  if len(activations) > 0:
    print(f"Saving {activations.shape} activations to {args.output}")
    np.save(args.output, activations)
  else:
    print("No activations collected (no episodes met the reward threshold).")

if __name__ == "__main__":
  main()
