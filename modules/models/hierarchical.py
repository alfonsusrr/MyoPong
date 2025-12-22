import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from modules.utils.predict_traj import predict_ball_trajectory

class HierarchicalTableTennisWrapper(gym.Wrapper):
    """
    A wrapper for TableTennisEnvV0 that augments the observation with 
    physics-based predictions following the logic in pong_v0.py.
    """
    def __init__(self, env):
        super().__init__(env)
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

        if hasattr(env_unwrapped, 'obs_dict'):
            obs_dict = env_unwrapped.obs_dict
            
            ball_pos = obs_dict.get('ball_pos')
            ball_vel = obs_dict.get('ball_vel')
            paddle_pos = obs_dict.get('paddle_pos')
            
            if ball_pos is not None and ball_vel is not None and paddle_pos is not None:
                # Use current paddle position (can be clamped to stable X if desired)
                stable_paddle_pos = paddle_pos.copy()
                
                # Use the refined physics-based utility function
                pred_ball_pos, paddle_ori_ideal = predict_ball_trajectory(
                    ball_pos, ball_vel, stable_paddle_pos,
                    gravity=self._gravity,
                    table_z=self._table_z,
                    ball_radius=self._ball_radius,
                    net_height=self._net_height
                )
                
                goal = np.concatenate([pred_ball_pos, paddle_ori_ideal])
                return np.concatenate([obs, goal]).astype(np.float32)
        
        return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Clear cached constants on reset in case model changed
        self._gravity = None 
        
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._augment_obs(obs), info
        return self._augment_obs(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._augment_obs(obs), reward, terminated, truncated, info
        elif len(result) == 4:
            obs, reward, done, info = result
            return self._augment_obs(obs), reward, done, info
        return result
