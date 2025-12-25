import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from modules.utils.predict_traj import predict_ball_trajectory

class HierarchicalTableTennisWrapper(gym.Wrapper):
    """
    A wrapper for TableTennisEnvV0 that augments the observation with 
    physics-based predictions and provides alignment rewards.
    """
    def __init__(self, env, update_freq=1, alignment_weights=None):
        super().__init__(env)
        self._update_freq = update_freq
        self._alignment_weights = alignment_weights or {
            "alignment_y": 0.5,
            "alignment_z": 0.5,
            "paddle_quat_goal": 0.5
        }

        self._step_count = 0
        # Augment the observation space: 
        # Original features + 3 (pred_ball_pos) + 4 (paddle_ori_ideal) = +7
        if hasattr(self.env.observation_space, 'shape'):
            old_shape = self.env.observation_space.shape[0]
            new_shape = (old_shape + 7,)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=new_shape, 
                dtype=np.float32
            )
        
        # Cache environment-specific constants for prediction
        self._gravity = None
        self._table_z = None
        self._ball_radius = None
        self._net_height = None
        self._frozen_goal = None
        self._cached_goal = None
        self._last_goal = None
        
        # Proper slew rate limits for different units
        # Position: max change in meters per step
        self._max_pos_delta = 0.05 
        # Orientation: max change in quaternion components per step
        self._max_rot_delta = 0.1

    def _get_env_constants(self):
        """Extract physical constants from the MuJoCo simulation."""
        env_unwrapped = self.env.unwrapped
        sim = env_unwrapped.sim
        model = sim.model
        data = sim.data
        id_info = env_unwrapped.id_info

        # Gravity (magnitude)
        self._gravity = float(-model.opt.gravity[2])
        if self._gravity <= 0: self._gravity = 9.81

        # Table Z (top of own half table)
        own_gid = id_info.own_half_gid
        table_top_z = float(data.geom_xpos[own_gid][2] + model.geom_size[own_gid][2])
        
        # Ball radius
        self._ball_radius = float(model.geom_size[id_info.ball_gid][0])
        
        # Combined table_z as used in predict_traj.py (contact height)
        self._table_z = table_top_z + self._ball_radius

        # Net height: table_height + net_half_height
        self._net_height = 0.95 

    def _augment_obs(self, obs):
        """Calculates the physics-based goal and appends it to the observation."""
        env_unwrapped = self.env.unwrapped
        
        if self._gravity is None:
            self._get_env_constants()

        obs_dict = getattr(env_unwrapped, 'obs_dict', None)
        if obs_dict is None:
            return obs

        # Fast extraction of needed values
        ball_pos = obs_dict.get('ball_pos')
        ball_vel = obs_dict.get('ball_vel')
        paddle_pos = obs_dict.get('paddle_pos')
        
        if ball_pos is None or ball_vel is None or paddle_pos is None:
            return obs

        ball_pos_x = ball_pos[0]
        
        # Determine if we should update the goal
        should_recalculate = (self._step_count % self._update_freq == 0) or (self._cached_goal is None)
        
        # Check freeze conditions early
        if self._frozen_goal is not None:
            ball_vel_x = ball_vel[0]
            paddle_pos_x = paddle_pos[0]
            touching = obs_dict.get('touching_info')
            paddle_touch = touching[0] if touching is not None else 0
            
            if (abs(ball_pos_x - paddle_pos_x) < 0.05) or (paddle_touch > 0) or (ball_vel_x < -0.05):
                goal = self._frozen_goal.copy()
            elif should_recalculate:
                pred_ball_pos, paddle_ori_ideal = predict_ball_trajectory(
                    ball_pos, ball_vel, paddle_pos,
                    gravity=self._gravity, table_z=self._table_z,
                    ball_radius=self._ball_radius, net_height=self._net_height
                )
                goal = np.concatenate([pred_ball_pos, paddle_ori_ideal])
                self._cached_goal = goal.copy()
                if ball_pos_x > 1.4:
                    self._frozen_goal = goal.copy()
            else:
                goal = self._cached_goal.copy()
        else:
            if should_recalculate:
                pred_ball_pos, paddle_ori_ideal = predict_ball_trajectory(
                    ball_pos, ball_vel, paddle_pos,
                    gravity=self._gravity, table_z=self._table_z,
                    ball_radius=self._ball_radius, net_height=self._net_height
                )
                goal = np.concatenate([pred_ball_pos, paddle_ori_ideal])
                self._cached_goal = goal.copy()
                if ball_pos_x > 1.4:
                    self._frozen_goal = goal.copy()
            else:
                goal = self._cached_goal.copy()
        
        # Apply Slew Rate Limiting
        if self._last_goal is not None:
            last_goal = self._last_goal
            
            # 1. Position Slew
            delta_pos = goal[:3] - last_goal[:3]
            dist_pos = np.linalg.norm(delta_pos)
            if dist_pos > self._max_pos_delta:
                goal[:3] = last_goal[:3] + (delta_pos * (self._max_pos_delta / (dist_pos + 1e-9)))
                
            # 2. Orientation Slew
            delta_ori = goal[3:] - last_goal[3:]
            dist_ori = np.linalg.norm(delta_ori)
            if dist_ori > self._max_rot_delta:
                curr_ori = last_goal[3:] + (delta_ori * (self._max_rot_delta / (dist_ori + 1e-9)))
                # Normalize
                goal[3:] = curr_ori / (np.linalg.norm(curr_ori) + 1e-9)
        
        self._last_goal = goal.copy()
        return np.concatenate([obs, goal]).astype(np.float32)

    def reset(self, **kwargs):
        self._step_count = 0
        self._cached_goal = None
        result = self.env.reset(**kwargs)
        # Clear cached constants on reset in case model changed
        self._gravity = None 
        self._frozen_goal = None
        self._last_goal = None
        
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._augment_obs(obs), info
        return self._augment_obs(result)

    def step(self, action):
        result = self.env.step(action)
        self._step_count += 1
        
        # Extract goal from current observation augmentation
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            aug_obs = self._augment_obs(obs)
            # Use the cached goal from _augment_obs
            goal = self._last_goal
            reward, info = self._augment_reward(reward, info, goal)
            return aug_obs, reward, terminated, truncated, info
        elif len(result) == 4:
            obs, reward, done, info = result
            aug_obs = self._augment_obs(obs)
            goal = self._last_goal
            reward, info = self._augment_reward(reward, info, goal)
            return aug_obs, reward, done, info
        return result

    def _augment_reward(self, reward, info, goal):
        """Calculates alignment rewards based on the goal and updates the scalar reward."""
        if 'rwd_dict' not in info or goal is None:
            return reward, info
            
        env_unwrapped = self.env.unwrapped
        obs_dict = getattr(env_unwrapped, 'obs_dict', None)
        if obs_dict is None:
            return reward, info

        # 1. Calculate masks
        # reach_err = paddle_pos - ball_pos
        err_x = obs_dict["reach_err"][0] 
        active_mask = float(err_x > -0.05)
        
        # Check if ball has hit paddle (PingpongContactLabels.PADDLE = 0)
        contact_trajectory = getattr(env_unwrapped, 'contact_trajectory', [])
        has_hit_paddle = any(any(getattr(item, 'value', item) == 0 for item in s) for s in contact_trajectory) 
        active_alignment_mask = active_mask * (1.0 - float(has_hit_paddle))

        # 2. Calculate alignment rewards
        pred_ball_pos = goal[:3]
        paddle_ori_goal = goal[3:]
        
        paddle_pos = obs_dict["paddle_pos"]
        pred_err_y = np.abs(paddle_pos[1] - pred_ball_pos[1])
        pred_err_z = np.abs(paddle_pos[2] - pred_ball_pos[2])
        
        alignment_y = active_alignment_mask * np.exp(-5.0 * pred_err_y)
        alignment_z = active_alignment_mask * np.exp(-5.0 * pred_err_z)
        
        paddle_ori = obs_dict["paddle_ori"]
        paddle_ori_err_goal = paddle_ori - paddle_ori_goal
        paddle_quat_err_goal = np.linalg.norm(paddle_ori_err_goal)
        paddle_quat_reward_goal = active_alignment_mask * np.exp(-5.0 * paddle_quat_err_goal)

        # 3. Add to rwd_dict and update scalar reward
        info['rwd_dict']["alignment_y"] = alignment_y
        info['rwd_dict']["alignment_z"] = alignment_z
        info['rwd_dict']["paddle_quat_goal"] = paddle_quat_reward_goal
        
        additional_reward = (
            alignment_y * self._alignment_weights["alignment_y"] +
            alignment_z * self._alignment_weights["alignment_z"] +
            paddle_quat_reward_goal * self._alignment_weights["paddle_quat_goal"]
        )
        
        return reward + additional_reward, info
