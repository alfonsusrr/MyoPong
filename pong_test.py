import warnings

# supress all warnings
warnings.filterwarnings("ignore")

import mujoco
import numpy as np
from myosuite.utils import gym
import os
import imageio

def test_reward():
    env_id = "myoChallengePongP0-v0"
    
    # Define randomization ranges matching tabletennis curriculum difficulty 3
    # These are the same as used in ppo-pong.py
    ball_xyz_range = {'high': [-0.8, 0.5, 1.5], 'low': [-1.25, -0.5, 1.4]}
    
    env = gym.make(env_id, ball_xyz_range=ball_xyz_range, ball_qvel=True)
    env.reset()
    
    sim = env.sim
    id_info = env.id_info
    
    # Print some info
    print(f"Action Space Low: {env.action_space.low}")
    print(f"Action Space High: {env.action_space.high}")
    print(f"Paddle Site ID: {id_info.paddle_sid}, Name: {sim.model.site(id_info.paddle_sid).name}")
    print(f"Paddle Geom ID: {id_info.paddle_gid}, Name: {sim.model.geom(id_info.paddle_gid).name}")
    print(f"Ball Site ID: {id_info.ball_sid}, Name: {sim.model.site(id_info.ball_sid).name}")
    
    # Get positions
    paddle_site_pos = sim.data.site_xpos[id_info.paddle_sid].copy()
    paddle_geom_pos = sim.data.geom_xpos[id_info.paddle_gid].copy()
    ball_pos = sim.data.site_xpos[id_info.ball_sid].copy()
    
    print(f"Initial Ball Pos: {ball_pos}")
    print(f"Initial Paddle Site Pos: {paddle_site_pos}")
    print(f"Initial Paddle Geom Pos: {paddle_geom_pos}")
    
    # Check reach_err calculation in env
    obs_dict = env.unwrapped.get_obs_dict(sim)
    reach_err = obs_dict['reach_err']
    print(f"Reach Err (paddle_pos - ball_pos): {reach_err}")
    
    # Test reward calculation logic
    err_x = reach_err[0]
    print(f"err_x (paddle_x - ball_x): {err_x}")
    
    active_mask = (err_x > 0)
    print(f"Active Mask (err_x > 0): {active_mask}")
    
    frames_cam1 = []
    frames_cam2 = []
    video_path_cam1 = "pong_test_render_cam1.mp4"
    video_path_cam2 = "pong_test_render_cam2.mp4"

    total_reward = 0
    # Let's step and see how it changes
    for i in range(200):
        # Capture frames from both cameras
        try:
            frame1 = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1)
            frames_cam1.append(frame1)
            frame2 = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=2)
            frames_cam2.append(frame2)
        except Exception as e:
            if i == 0:
                print(f"Warning: Rendering failed: {e}")

        # Move the paddle toward the net (X) and side (Y)
        # With normalize_act=True, valid ranges are [-1, 1] for all axes.
        # -1 maps to min ctrlrange, 1 maps to max ctrlrange.
        # Example: action = [0, 0, 0] is the center of the workspace.
        action = np.array([-0.5, 0.7, 0.6, 0.0, 0.0, 0.0])
        
        step_results = env.step(action)
        if len(step_results) == 5:
            obs_vec, rwd, terminated, truncated, info = step_results
            done = terminated or truncated
        else:
            obs_vec, rwd, done, info = step_results
        
        # Get the named observations from the unwrapped environment
        obs_dict = env.unwrapped.get_obs_dict(env.sim)
        curr_ball_pos = obs_dict['ball_pos']
        curr_paddle_pos = obs_dict['paddle_pos']
        ori_err_norm = np.linalg.norm(obs_dict['paddle_ori_err'])
        
        print(f"Step {i+1}: Ball: {curr_ball_pos}, Paddle: {curr_paddle_pos}, Ori Err: {ori_err_norm:.4f}, Reward: {rwd:.4f}")
        total_reward += rwd
        if done:
            print(f"Done at step {i+1}")
            break
            
    if frames_cam1:
        print(f"Saving video to {video_path_cam1}...")
        imageio.mimwrite(video_path_cam1, frames_cam1, fps=30)
        print("Video 1 saved.")
    
    if frames_cam2:
        print(f"Saving video to {video_path_cam2}...")
        imageio.mimwrite(video_path_cam2, frames_cam2, fps=30)
        print("Video 2 saved.")
    
    if not frames_cam1 and not frames_cam2:
        print("No frames captured.")

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_reward()
