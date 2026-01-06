import warnings

# suppress all warnings
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
    max_delta = 0.1
    
    if last_actions is not None:
        delta = target_actions - last_actions
        actions = last_actions + np.clip(delta, -max_delta, max_delta)
    else:
        actions = target_actions
        
    return actions

def sample_trajectories():
    env_id = "myoChallengePongP0-v0"
    
    # Define randomization ranges matching tabletennis curriculum difficulty 3
    ball_xyz_range = {'high': [-0.8, 0.5, 1.5], 'low': [-1.25, -0.5, 1.4]}
    
    env = gym.make(env_id, ball_xyz_range=ball_xyz_range, ball_qvel=True)
    
    # Get actuator control range for normalization
    ctrl_range = env.sim.model.actuator_ctrlrange
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    
    output_dir = "pong_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = 10
    
    for i in range(num_samples):
        print(f"Starting sample {i+1}/{num_samples}...")
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            
        last_actions = np.zeros((1, 6))
        frozen_actions = [None]
        frames = []
        
        # Max steps per rollout
        for step in range(500):
            obs_vec = obs.reshape(1, -1)
            actions = heuristic_policy(obs_vec, ctrl_min, ctrl_max, last_actions, frozen_actions)
            last_actions = actions.copy()
            action = actions[0]
            
            step_results = env.step(action)
            if len(step_results) == 5:
                obs, rwd, terminated, truncated, info = step_results
                done = terminated or truncated
            else:
                obs, rwd, done, info = step_results
            
            # Render camera 3 (overview)
            frame = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=3)
            frames.append(frame)
            
            if done:
                break
        
        if frames:
            video_path = os.path.join(output_dir, f"sample_{i+1}_cam3.mp4")
            imageio.mimwrite(video_path, frames, fps=30)
            print(f"  Sample {i+1} saved to {video_path} ({len(frames)} steps)")
        else:
            print(f"  Sample {i+1} failed to capture any frames.")

    env.close()
    print("\nAll samples completed.")

if __name__ == "__main__":
    sample_trajectories()

