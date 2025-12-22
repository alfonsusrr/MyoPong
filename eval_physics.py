import numpy as np
import os
import time
import argparse
import warnings
import imageio
from typing import Dict, Any, List
from tqdm import tqdm

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

def heuristic_policy(obs_vec, ctrl_min, ctrl_max, last_actions=None, alpha=0.2):
    """
    Heuristic policy logic from pong_test.py
    obs_vec: (N_envs, 32)
    """
    # Indices based on DEFAULT_OBS_KEYS in pong_v0.py:
    # 0:3   ball_pos
    # 3:6   ball_vel
    # 6:9   paddle_pos
    # 9:12  paddle_vel
    # 12:16 paddle_ori (quat)
    # 16:19 reach_err
    # 19:25 touching_info (6)
    # 25:28 pred_ball_pos
    # 28:32 paddle_ori_ideal (quat)
    
    # Extract predicted ball position and ideal paddle orientation from observations
    # Note: These are already calculated by the environment's get_obs_dict
    pred_ball_pos = obs_vec[:, 25:28]
    paddle_ori_ideal = obs_vec[:, 28:32]
    
    # Paddle offset in body frame (calculated from check_paddle.py)
    paddle_offset_body_frame = np.array([-5.5743e-05, 2.0686e-02, 6.6874e-02])
    
    # Position Control
    # base_pos matches the one in pong_test.py
    base_pos = np.array([1.8, 0.5, 1.13])
    
    actions_pos = []
    actions_rot = []
    pos_min = ctrl_min[:3]
    pos_max = ctrl_max[:3]
    rot_min = ctrl_min[3:6]
    rot_max = ctrl_max[3:6]
    
    for i in range(obs_vec.shape[0]):
        # Account for offset and rotation
        # paddle_ori_ideal is [w, x, y, z] in MyoSuite
        # R.from_quat expects [x, y, z, w]
        q = paddle_ori_ideal[i]
        r_ideal = R.from_quat([q[1], q[2], q[3], q[0]])
        rotated_offset = r_ideal.apply(paddle_offset_body_frame)
        
        # Desired joint positions (relative to base)
        # joint_pos = pred_ball_pos - base_pos - offset
        desired_joint_pos = pred_ball_pos[i] - base_pos - rotated_offset
        
        # Normalize position
        action_pos = 2.0 * (desired_joint_pos - pos_min) / (pos_max - pos_min) - 1.0
        action_pos = np.clip(action_pos, -1.0, 1.0)
        actions_pos.append(action_pos)
        
        # Orientation Control
        desired_euler = r_ideal.as_euler('xyz')
        
        # Normalize rotation
        action_rot = 2.0 * (desired_euler - rot_min) / (rot_max - rot_min) - 1.0
        action_rot = np.clip(action_rot, -1.0, 1.0)
        actions_rot.append(action_rot)
    
    actions_pos = np.array(actions_pos)
    actions_rot = np.array(actions_rot)
    
    target_actions = np.concatenate([actions_pos, actions_rot], axis=-1)
    
    # Smoothing
    if last_actions is None:
        actions = target_actions
    else:
        actions = alpha * target_actions + (1.0 - alpha) * last_actions
        
    return actions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Physics Heuristic Policy")
    parser.add_argument("--env-id", type=str, default="myoChallengePongP0-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=100, help="Total number of episodes")
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
        actions = heuristic_policy(obs, ctrl_min, ctrl_max, last_actions=last_actions)
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

    pbar.close()
    envs.close()

    # Calculate and print metrics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = np.mean(all_successes) * 100
    std_success = np.std(all_successes) * 100
    hit_paddle_rate = np.mean(all_hit_paddles) * 100
    std_hit_paddle = np.std(all_hit_paddles) * 100

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

