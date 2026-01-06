import warnings

# supress all warnings
warnings.filterwarnings("ignore")

import mujoco
import numpy as np
from myosuite.utils import gym
import os
import imageio
from scipy.spatial.transform import Rotation as R

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

def test_reward():
    env_id = "myoChallengePongP0-v0"
    
    # Define randomization ranges matching tabletennis curriculum difficulty 3
    # These are the same as used in ppo-pong.py
    ball_xyz_range = {'high': [-0.8, 0.5, 1.5], 'low': [-1.25, -0.5, 1.4]}
    
    env = gym.make(env_id, ball_xyz_range=ball_xyz_range, ball_qvel=True)
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
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
    last_actions = np.zeros((1, 6))
    frozen_actions = [None]
    
    # Let's step and see how it changes
    for i in range(200):
        # Use heuristic policy from eval_physics.py
        # obs is the vector from env.reset() or env.step()
        obs_vec = obs.reshape(1, -1)
        actions = heuristic_policy(obs_vec, ctrl_min, ctrl_max, last_actions, frozen_actions)
        last_actions = actions.copy()
        action = actions[0]
        
        # Step environment
        step_results = env.step(action)
        print(f"Sim Ctrl (after step): {env.sim.data.ctrl[:6]}")
        if len(step_results) == 5:
            obs, rwd, terminated, truncated, info = step_results
            done = terminated or truncated
        else:
            obs, rwd, done, info = step_results
        
        # Get the named observations from the unwrapped environment for logging
        obs_dict = env.unwrapped.get_obs_dict(env.sim)
        curr_ball_pos = obs_dict['ball_pos']
        curr_paddle_pos = obs_dict['paddle_pos']
        curr_ball_vel = obs_dict['ball_vel']
        touching_info = obs_dict['touching_info']
        pred_ball_pos = obs_dict['pred_ball_pos']
        paddle_ori_ideal = obs_dict['paddle_ori_ideal']
        
        # --- COMPARISON START (predict_traj vs env) ---
        from modules.utils.predict_traj import predict_ball_trajectory
        g = float(-env.sim.model.opt.gravity[2])
        own_gid = env.unwrapped.id_info.own_half_gid
        table_z_plane = float(env.sim.data.geom_xpos[own_gid][2] + env.sim.model.geom_size[own_gid][2])
        ball_r = float(env.sim.model.geom_size[env.unwrapped.id_info.ball_gid][0])
        z_contact = table_z_plane + ball_r
        
        # stable_paddle_pos as used in environment's get_obs_dict
        stable_paddle_pos = curr_paddle_pos.copy()
        stable_paddle_pos[0] = 1.62
        
        pred_ball_pos_2, paddle_ori_ideal_2 = predict_ball_trajectory(
            curr_ball_pos, curr_ball_vel, stable_paddle_pos,
            gravity=g,
            table_z=z_contact,
            ball_radius=ball_r
        )
        
        pos_diff = np.linalg.norm(pred_ball_pos - pred_ball_pos_2)
        ori_diff = np.linalg.norm(paddle_ori_ideal - paddle_ori_ideal_2)
        if pos_diff > 1e-6 or ori_diff > 1e-6:
            print(f"!!! COMPARISON MISMATCH at step {i+1} !!!")
            print(f"  Pos diff: {pos_diff:.8f}")
            print(f"  Ori diff: {ori_diff:.8f}")
        # --- COMPARISON END ---

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
        print(f"  Ball Vel: {curr_ball_vel}")
        print(f"  Paddle Pos: {curr_paddle_pos}")
        print(f"  Paddle Ori (Quat): {paddle_ori}")
        print(f"  Pred Ball Pos: {pred_ball_pos}")
        print(f"  Step Reward: {rwd:.4f}")

        # frame capture...

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
