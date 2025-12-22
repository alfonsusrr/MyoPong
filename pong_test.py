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
    
    from scipy.spatial.transform import Rotation as R

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
    frames_cam3 = []
    video_path_cam1 = "pong_test_render_cam1.mp4"
    video_path_cam2 = "pong_test_render_cam2.mp4"
    video_path_cam3 = "pong_test_render_cam3.mp4"
    
    # Get actuator control range for normalization
    ctrl_range = env.sim.model.actuator_ctrlrange
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    print(f"Ctrl Min: {ctrl_min}")
    print(f"Ctrl Max: {ctrl_max}")

    total_reward = 0
    last_action = None
    alpha = 0.5  # Smoothing factor
    locked_impact = None
    
    # Calculate paddle offset once
    # Body Pos (joints=0): [1.8  0.5  1.13]
    # Geom Pos (joints=0): [1.7999 0.5207 1.1969]
    # We'll use the values from check_paddle.py for precision
    paddle_offset_body_frame = np.array([-5.5743e-05, 2.0686e-02, 6.6874e-02])

    # Let's step and see how it changes
    for i in range(200):
        # Get current state for prediction
        obs_dict = env.unwrapped.get_obs_dict(env.sim)
        curr_ball_pos = obs_dict['ball_pos']
        curr_ball_vel = obs_dict['ball_vel']
        curr_paddle_pos = obs_dict['paddle_pos']
        curr_paddle_vel = obs_dict['paddle_vel']
        
        # Fixed impact plane for stability
        target_x = 1.62
        
        # Estimate time to impact
        vx = curr_ball_vel[0]
        dx = target_x - curr_ball_pos[0]
        dt = dx / vx if vx > 0.1 else 2.0
        
        if dt < 0.08 and locked_impact is not None:
            # Lock prediction when very close to impact to avoid jitters
            pred_ball_pos, n_ideal, paddle_ori_ideal = locked_impact
        else:
            # Use stable X-plane for prediction
            stable_paddle_pos = curr_paddle_pos.copy()
            stable_paddle_pos[0] = target_x
            
            pred_ball_pos, n_ideal, paddle_ori_ideal = env.unwrapped.calculate_prediction(
                curr_ball_pos, curr_ball_vel, stable_paddle_pos, curr_paddle_vel
            )
            
            if dt < 0.15:
                locked_impact = (pred_ball_pos, n_ideal, paddle_ori_ideal)

        # Calculate desired action based on prediction
        base_pos = np.array([1.8, 0.5, 1.13])
        
        # Account for offset and rotation
        r_ideal = R.from_quat([paddle_ori_ideal[1], paddle_ori_ideal[2], paddle_ori_ideal[3], paddle_ori_ideal[0]])
        rotated_offset = r_ideal.apply(paddle_offset_body_frame)
        
        # Desired joint positions (relative to base)
        # joint_pos = pred_ball_pos - base_pos - rotated_offset
        desired_joint_pos = pred_ball_pos - base_pos - rotated_offset
        
        # Normalize to [-1, 1] using the actuator control ranges
        pos_min = ctrl_min[:3]
        pos_max = ctrl_max[:3]
        action_pos = 2.0 * (desired_joint_pos - pos_min) / (pos_max - pos_min) - 1.0
        action_pos = np.clip(action_pos, -1.0, 1.0)
        
        # Calculate ideal orientation (Euler angles)
        desired_euler = r_ideal.as_euler('xyz')
        
        rot_min = ctrl_min[3:6]
        rot_max = ctrl_max[3:6]
        action_rot = 2.0 * (desired_euler - rot_min) / (rot_max - rot_min) - 1.0
        action_rot = np.clip(action_rot, -1.0, 1.0)
        
        # Combine position and rotation actions
        target_action = np.concatenate([action_pos, action_rot])
        
        # Smoothing
        if last_action is None:
            action = target_action
        else:
            action = alpha * target_action + (1.0 - alpha) * last_action
        
        last_action = action.copy()
        
        # Step environment
        step_results = env.step(action)
        print(f"Sim Ctrl (after step): {env.sim.data.ctrl[:6]}")
        if len(step_results) == 5:
            obs_vec, rwd, terminated, truncated, info = step_results
            done = terminated or truncated
        else:
            obs_vec, rwd, done, info = step_results
        
        # Get the named observations from the unwrapped environment
        obs_dict = env.unwrapped.get_obs_dict(env.sim)
        curr_ball_pos = obs_dict['ball_pos']
        curr_paddle_pos = obs_dict['paddle_pos']
        ball_vel = obs_dict["ball_vel"]
        touching_info = obs_dict["touching_info"]
        
        # touching_info is [paddle, own, opponent, net, ground, env]
        hit_paddle = touching_info[0] > 0
        if hit_paddle:
            print(f"  *** BALL HIT PADDLE at step {i+1} ***")
        hit_opponent = touching_info[2] > 0
        if hit_opponent:
            print(f"  *** BALL HIT OPPONENT SIDE at step {i+1} ***")
        
        # Calculate n_curr manually to see where it's pointing
        paddle_ori = obs_dict["paddle_ori"]
        
        print(f"Step {i+1}:")
        print(f"  Ball Pos: {curr_ball_pos}")
        print(f"  Ball Vel: {ball_vel}")
        print(f"  Paddle Pos: {curr_paddle_pos}")
        print(f"  Paddle Ori (Quat): {paddle_ori}")
        print(f"  Pred Ball Pos: {pred_ball_pos}")
        print(f"  Step Reward: {rwd:.4f}")

        # rwd_dict = env.unwrapped.get_reward_dict(obs_dict)

        # for key, value in rwd_dict.items():
        #     print(f"  {key}: {np.array(value).squeeze():.4f}", end= " | ")
        # print()

        frame_cam1 = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1)
        frames_cam1.append(frame_cam1)
        frame_cam2 = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=2)
        frames_cam2.append(frame_cam2)
        frame_cam3 = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=3)
        frames_cam3.append(frame_cam3)

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

    if frames_cam3:
        print(f"Saving video to {video_path_cam3}...")
        imageio.mimwrite(video_path_cam3, frames_cam3, fps=30)
        print("Video 3 saved.")
    
    if not frames_cam1 and not frames_cam2 and not frames_cam3:
        print("No frames captured.")

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_reward()
