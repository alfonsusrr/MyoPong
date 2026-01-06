import numpy as np
import os
import sys
import time
import argparse
import warnings
import imageio
from typing import Dict, Any, List
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from myosuite.utils import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.envs.curriculum import tabletennis_curriculum_kwargs
from scipy.spatial.transform import Rotation as R

# Suppress warnings
warnings.filterwarnings("ignore")

def make_env(env_id: str, difficulty: int, seed: int):
    def _init():
        kwargs = tabletennis_curriculum_kwargs(difficulty=difficulty)
        # Ensure we get the full observation vector needed for the heuristic
        env = gym.make(env_id, **kwargs)
        
        # Add a helper method for offscreen rendering that can be called via env_method
        def render_offscreen(width=640, height=480, camera_id=1):
            return env.sim.renderer.render_offscreen(width=width, height=height, camera_id=camera_id)
        
        # Attach it to the env instance
        env.render_offscreen = render_offscreen
        
        return env
    return _init

def heuristic_policy(obs_vec, ctrl_min, ctrl_max, last_actions, frozen_actions):
    """
    Improved heuristic policy with slew rate limiting and impact stabilization.
    obs_vec: (N_envs, 32)
    """
    # 0:3 ball_pos, 3:6 ball_vel, 6:9 paddle_pos, 9:12 paddle_vel
    ball_pos_x = obs_vec[:, 0]
    ball_vel_x = obs_vec[:, 3]
    paddle_touch = obs_vec[:, 19]

    paddle_pos_x = obs_vec[:, 6]
    
    # Extract pre-calculated targets from observations
    pred_ball_pos = obs_vec[:, 25:28]
    paddle_ori_ideal = obs_vec[:, 28:32]
    
    # Paddle offset in body frame
    paddle_offset_body_frame = np.array([-5.5743e-05, 2.0686e-02, 6.6874e-02])
    base_pos = np.array([1.8, 0.5, 1.13])
    
    pos_min, pos_max = ctrl_min[:3], ctrl_max[:3]
    rot_min, rot_max = ctrl_min[3:6], ctrl_max[3:6]
    
    target_actions = np.zeros((obs_vec.shape[0], 6))
    
    for i in range(obs_vec.shape[0]):
        # Stabilization logic:
        # 1. Freeze target if ball is very close to impact (x > 1.5)
        # 2. Freeze target if paddle is already touching the ball
        # 3. Freeze target if ball is moving away (hit already happened)
        should_freeze = (abs(ball_pos_x[i] - paddle_pos_x[i]) < 0.05) or (paddle_touch[i] > 0) or (ball_vel_x[i] < -0.05)
        
        if should_freeze and frozen_actions[i] is not None:
            target_actions[i] = frozen_actions[i]
        else:
            # Calculate new target action
            q = paddle_ori_ideal[i]
            r_ideal = R.from_quat([q[1], q[2], q[3], q[0]])
            rotated_offset = r_ideal.apply(paddle_offset_body_frame)
            
            desired_joint_pos = pred_ball_pos[i] - base_pos - rotated_offset
            action_pos = 2.0 * (desired_joint_pos - pos_min) / (pos_max - pos_min) - 1.0
            
            desired_euler = r_ideal.as_euler('xyz')
            action_rot = 2.0 * (desired_euler - rot_min) / (rot_max - rot_min) - 1.0
            
            new_action = np.concatenate([action_pos, action_rot])
            target_actions[i] = np.clip(new_action, -1.0, 1.0)
            
            # Update frozen action if we are approaching the freeze zone
            if ball_pos_x[i] > 1.4:
                frozen_actions[i] = target_actions[i].copy()

    # Slew Rate Limiting: limit max change per step to prevent "spinning" and instability
    # Max change per step in normalized [-1, 1] units
    # 0.1 allows full range travel in ~20 steps (0.2 seconds at 100Hz)
    max_delta = 0.1
    
    if last_actions is not None:
        delta = target_actions - last_actions
        actions = last_actions + np.clip(delta, -max_delta, max_delta)
    else:
        actions = target_actions
        
    return actions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Physics Heuristic Policy")
    parser.add_argument("--env-id", type=str, default="myoChallengePongP0-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=400, help="Total number of episodes")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--difficulty", type=int, default=5, help="Curriculum difficulty (0-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render-fails", action="store_true", help="Render and save failed episodes")
    parser.add_argument("--video-dir", type=str, default="failed_episodes", help="Directory to save failed episode videos")
    args = parser.parse_args()

    if args.render_fails:
        os.makedirs(args.video_dir, exist_ok=True)

    print(f"Starting evaluation of physics heuristic policy on {args.env_id}")
    print(f"Difficulty: {args.difficulty}, Episodes: {args.num_episodes}, Parallel Envs: {args.num_envs}")

    # Create parallel environments
    env_fns = [make_env(args.env_id, args.difficulty, args.seed + i) for i in range(args.num_envs)]
    envs = SubprocVecEnv(env_fns)

    # Retrieve control ranges from a temporary local environment to avoid pickling issues with 'sim'
    temp_env = gym.make(args.env_id, **tabletennis_curriculum_kwargs(difficulty=args.difficulty))
    ctrl_ranges = temp_env.sim.model.actuator_ctrlrange.copy()
    ctrl_min = ctrl_ranges[:, 0]
    ctrl_max = ctrl_ranges[:, 1]
    temp_env.close()

    obs = envs.reset()
    last_actions = np.zeros((args.num_envs, 6))
    frozen_actions = [None] * args.num_envs
    
    episodes_completed = 0
    all_rewards = []
    all_successes = []
    all_hit_paddles = []
    
    current_episode_rewards = np.zeros(args.num_envs)
    current_episode_success = np.zeros(args.num_envs, dtype=bool)
    current_episode_hit_paddle = np.zeros(args.num_envs, dtype=bool)
    
    # Store frames for each environment's current episode
    env_frames = [[] for _ in range(args.num_envs)]

    pbar = tqdm(total=args.num_episodes)

    while episodes_completed < args.num_episodes:
        # Get heuristic action
        actions = heuristic_policy(obs, ctrl_min, ctrl_max, last_actions, frozen_actions)
        last_actions = actions.copy()
        
        # Capture frames before step if rendering is enabled
        if args.render_fails:
            frames = envs.env_method("render_offscreen", width=640, height=480, camera_id=3)
            for i in range(args.num_envs):
                env_frames[i].append(frames[i])
        
        # Step environments
        obs, rewards, dones, infos = envs.step(actions)
        
        current_episode_rewards += rewards
        
        for i in range(args.num_envs):
            # Check for success/hit in this step
            if 'rwd_dict' in infos[i]:
                rwd_dict = infos[i]['rwd_dict']
                
                # Check for success
                solved = rwd_dict.get('solved', False)
                if isinstance(solved, np.ndarray):
                    solved = np.any(solved)
                if solved:
                    current_episode_success[i] = True
                
                # Check for paddle hit
                sparse = rwd_dict.get('sparse', 0)
                if isinstance(sparse, np.ndarray):
                    sparse = np.max(sparse)
                if sparse > 0:
                    current_episode_hit_paddle[i] = True
            
            if dones[i]:
                if episodes_completed < args.num_episodes:
                    all_rewards.append(current_episode_rewards[i])
                    all_successes.append(current_episode_success[i])
                    all_hit_paddles.append(current_episode_hit_paddle[i])
                    
                    # Save video if the episode failed and rendering is enabled
                    if args.render_fails and not current_episode_success[i]:
                        video_path = os.path.join(args.video_dir, f"fail_ep_{episodes_completed}_env_{i}.mp4")
                        if len(env_frames[i]) > 0:
                            imageio.mimwrite(video_path, env_frames[i], fps=30)
                    
                    episodes_completed += 1
                    pbar.update(1)
                
                # Reset tracking for this env slot
                current_episode_rewards[i] = 0
                current_episode_success[i] = False
                current_episode_hit_paddle[i] = False
                env_frames[i] = []
                last_actions[i] = 0 # Reset action for new episode
                frozen_actions[i] = None

    pbar.close()
    envs.close()

    # Calculate and print metrics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = np.mean(all_successes) * 100
    std_success = np.std(all_successes) * 100 / np.sqrt(len(all_successes))
    hit_paddle_rate = np.mean(all_hit_paddles) * 100
    std_hit_paddle = np.std(all_hit_paddles) * 100 / np.sqrt(len(all_hit_paddles))

    print("\n" + "="*30)
    print("EVALUATION METRICS (Physics Heuristic)")
    print("="*30)
    print(f"Total Episodes:   {len(all_rewards)}")
    print(f"Average Reward:   {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate:     {success_rate:.2f}% +/- {std_success:.2f}%")
    print(f"Paddle Hit Rate:  {hit_paddle_rate:.2f}% +/- {std_hit_paddle:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()

