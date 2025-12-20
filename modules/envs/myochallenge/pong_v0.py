""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Cheryl Wang (cheryl.wang.huiyi@gmail.com), Balint Hodossy (bkh16@ic.ac.uk), 
            Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================= """

import collections
from typing import List
import enum

import mujoco
import numpy as np
from myosuite.utils import gym
import os

from myosuite.utils.quat_math import euler2quat
from myosuite.envs import env_base


MAX_TIME = 3.0


class PongEnvV0(env_base.MujocoEnv):
    """
    Simplified Pong Environment.
    Inherits directly from MujocoEnv to bypass BaseV0's muscle and body logic.
    """

    MYO_CREDIT = """\
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """

    DEFAULT_OBS_KEYS = ['ball_pos', 'ball_vel', 'paddle_pos', "paddle_vel", 'paddle_ori', 'reach_err' , "touching_info"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": 5.0,
        "alignment_y": 10,
        "alignment_z": 10,
        "paddle_quat": 5,
        "move_reg": 0.01,
        "act_reg": 0.1,
        "sparse": 100,
        "solved": 1000,
        'done': -10
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        
        # Load the model directly without complex preprocessing
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip: int = 10,
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            ball_xyz_range = None,
            ball_qvel = None,
            ball_friction_range = None,
            paddle_mass_range = None,
            rally_count = 1,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):

        self.ball_xyz_range = ball_xyz_range
        self.ball_qvel = ball_qvel
        self.qpos_noise_range = qpos_noise_range
        self.paddle_mass_range = paddle_mass_range
        self.ball_friction_range = ball_friction_range
        
        # Default paddle orientation (identity in the new model as tilt is in visual child body)
        self.init_paddle_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.contact_trajectory = []

        self.id_info = IdInfo(self.sim.model)
        self.ball_dofadr = self.sim.model.body_dofadr[self.id_info.ball_bid]
        self.ball_posadr = self.sim.model.joint("pingpong_freejoint").qposadr[0]
        # In impedance mode, paddle uses 6 joints (slide+hinge)
        self.paddle_dofadr = self.sim.model.jnt_dofadr[self.sim.model.joint("paddle_x").id]
        self.paddle_posadr = self.sim.model.jnt_qposadr[self.sim.model.joint("paddle_x").id]

        # Call env_base.MujocoEnv._setup directly to avoid BaseV0 muscle logic
        # Remove keys that are explicitly passed to avoid "multiple values for keyword argument" error
        kwargs.pop('normalize_act', None)
        kwargs.pop('frame_skip', None)

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    normalize_act=False,
                    **kwargs,
        )

        # Reset camera/viewer if needed
        self.viewer_setup(azimuth=90, distance=1.5, render_actuator=False)

        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.start_vel = np.array([[5.6, 1.6, 0.1] ]) 
        self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel
        self.rally_count = rally_count
        self.cur_rally = 0

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        # Core paddle and ball observations
        obs_dict["ball_pos"] = sim.data.site_xpos[self.id_info.ball_sid].copy()
        obs_dict["ball_vel"] = self.get_sensor_by_name(sim.model, sim.data, "pingpong_vel_sensor").copy()

        obs_dict["paddle_pos"] = sim.data.site_xpos[self.id_info.paddle_sid].copy()
        obs_dict["paddle_vel"] = self.get_sensor_by_name(sim.model, sim.data, "paddle_vel_sensor").copy()
        obs_dict["paddle_ori"] = sim.data.body_xquat[self.id_info.paddle_bid].copy()
        obs_dict['padde_ori_err'] = obs_dict["paddle_ori"] - self.init_paddle_quat

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

        # Contact tracking for rewards and termination
        this_model = sim.model
        this_data = sim.data

        touching_objects = set(get_ball_contact_labels(this_model, this_data, self.id_info))
        self.contact_trajectory.append(touching_objects)

        obs_vec = self._ball_label_to_obs(touching_objects)
        obs_dict["touching_info"] = obs_vec

        # Actuator state if applicable
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_err = self.obs_dict['reach_err']
        reach_dist = np.linalg.norm(reach_err, axis=-1)

        # (2) Alignment reward: YZ error weighted by X closeness
        # Height (Z) and Left-Right (Y) alignment
        err_x = reach_err[..., 0] # paddle_x - ball_x
        err_y = np.abs(reach_err[..., 1])
        err_z = np.abs(reach_err[..., 2])
        
        # Closer ball = higher weight for YZ alignment
        alignment_w = 1.0 / (1.0 + 2 * np.abs(err_x))
        
        # Mask for ball passing paddle (no reward after ball passes paddle)
        active_mask = (err_x < 0).astype(float)
        
        # Check if ball has hit our table
        has_bounced = any(PingpongContactLabels.OWN in s for s in self.contact_trajectory)
        
        alignment_y = active_mask * alignment_w * np.exp(-5.0 * err_y)
        alignment_z = active_mask * alignment_w * np.exp(-5.0 * err_z) if has_bounced else 0.0

        # (1) Stability rewards
        paddle_vel = np.linalg.norm(obs_dict['paddle_vel'], axis=-1)

        # Angular velocity from sim data (handle potential batch dimension)
        paddle_ang_vel_raw = self.sim.data.qvel[self.paddle_dofadr+3 : self.paddle_dofadr+6]
        paddle_ang_vel = np.linalg.norm(paddle_ang_vel_raw)
        
        # Action Regularization (energy usage)
        # Using control input if available
        act_mag = np.linalg.norm(self.sim.data.ctrl, axis=-1) if self.sim.model.na > 0 else 0.0

        ball_pos = obs_dict["ball_pos"]
        solved = evaluate_pingpong_trajectory(self.contact_trajectory) == None
        paddle_quat_err = np.linalg.norm(obs_dict['padde_ori_err'], axis=-1)
        paddle_touch = obs_dict['touching_info'][..., 0]

        rwd_dict = collections.OrderedDict((
            ('reach_dist', active_mask * np.exp(-1. * reach_dist)),
            ('alignment_y', alignment_y),
            ('alignment_z', alignment_z),
            ('paddle_quat', np.exp(-5. * paddle_quat_err)),
            ('move_reg', -1.*(0.1 * paddle_vel + 0.1 * paddle_ang_vel)), # Reduced velocity penalty
            ('act_reg', -1. * act_mag),
            ('sparse', np.array([paddle_touch > 0])), 
            ('solved', np.array([[solved]])),
            ('done', np.array([[self._get_done(ball_pos[..., 2], solved)]])),
        ))

        rwd_dict['dense'] = sum(float(wt) * float(np.array(rwd_dict[key]).squeeze())
                            for key, wt in self.rwd_keys_wt.items()
                                if key in rwd_dict) # Added check for keys


        if rwd_dict['solved']:
            self.cur_rally += 1
        if rwd_dict['solved'] and self.cur_rally < self.rally_count:
            rwd_dict['done'] = False
            rwd_dict['solved'] = False
            # Ensure we update the batched time correctly
            self.obs_dict['time'].fill(0)
            self.sim.data.time = 0
            self.contact_trajectory = []
            self.relaunch_ball()
        return rwd_dict
    
    def _get_done(self, z, solved):
        # Handle potential batched Z-coordinate
        z_scalar = np.min(z)
        if np.any(self.obs_dict['time'] > MAX_TIME):
            return 1
        elif z_scalar < 0.3:
            self.obs_dict['time'].fill(MAX_TIME)
            return 1
        elif solved:
            return 1
        elif evaluate_pingpong_trajectory(self.contact_trajectory) in [0, 2, 3]:
            return 1
        return 0

    def _ball_label_to_obs(self, touching_body):
        obs_vec = np.array([0, 0, 0, 0, 0, 0])
        for i in touching_body:
            if i == PingpongContactLabels.PADDLE:
                obs_vec[0] += 1
            elif i == PingpongContactLabels.OWN:
                obs_vec[1] += 1
            elif i == PingpongContactLabels.OPPONENT:
                obs_vec[2] += 1
            elif i == PingpongContactLabels.NET:
                obs_vec[3] += 1
            elif i == PingpongContactLabels.GROUND:
                obs_vec[4] += 1
            else:
                obs_vec[5] += 1
        return obs_vec


    def get_metrics(self, paths, successful_steps=1):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) >= successful_steps:
                num_success += 1
        score = num_success/num_paths
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])
        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics
    
    def get_sensor_by_name(self, model, data, name):
        sensor_id = model.sensor_name2id(name)
        start = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        return data.sensordata[start:start+dim]


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.contact_trajectory = []
        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()

        if self.paddle_mass_range:
            self.sim.model.body_mass[self.id_info.paddle_bid] = self.np_random.uniform(*self.paddle_mass_range) 

        if self.ball_friction_range:
            self.sim.model.geom_friction[self.id_info.ball_gid] = self.np_random.uniform(**self.ball_friction_range)

        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.sim.model.body_pos[self.id_info.ball_bid] = ball_pos
            self.init_qpos[self.ball_posadr : self.ball_posadr + 3] = ball_pos
        else:
            ball_pos = self.init_qpos[self.ball_posadr : self.ball_posadr + 3]
        
        if self.qpos_noise_range is not None:
            # Randomize paddle joints (first 6)
            joint_ranges = self.sim.model.jnt_range[self.paddle_dofadr : self.paddle_dofadr+6, 1] - self.sim.model.jnt_range[self.paddle_dofadr : self.paddle_dofadr+6, 0]
            noise_fraction = self.np_random.uniform(**self.qpos_noise_range, size=(6,))
            reset_qpos_local = self.init_qpos.copy()
            for j in range(6):
                adr = self.sim.model.jnt_qposadr[self.paddle_dofadr + j]
                reset_qpos_local[adr] += noise_fraction[j] * joint_ranges[j]
                reset_qpos_local[adr] = np.clip(reset_qpos_local[adr], self.sim.model.jnt_range[self.paddle_dofadr + j, 0], self.sim.model.jnt_range[self.paddle_dofadr + j, 1])
        else:
            reset_qpos_local = reset_qpos if reset_qpos is not None else self.init_qpos

        if self.ball_qvel:            
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = ball_vel
        
        # Call MujocoEnv.reset directly to avoid BaseV0 muscle logic
        obs = super().reset(reset_qpos=reset_qpos_local, reset_qvel=self.init_qvel,**kwargs)
        self.cur_rally = 0
        return obs

    def cal_ball_qvel(self, ball_qpos):
        table_upper = [1.35, 0.70, 0.785] 
        table_lower = [0.5, -0.60, 0.785]
        gravity = 9.81
        v_z = self.np_random.uniform(*(-0.1, 0.1))

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            discriminant = 0
        t = (-b - discriminant**0.5) / (2 * a)

        v_upper = [(table_upper[i] - ball_qpos[i]) / t for i in range(2)]
        v_lower = [(table_lower[i] - ball_qpos[i]) / t for i in range(2)]

        return [[v_upper[0], v_upper[1], v_z], [v_lower[0], v_lower[1], v_z]]

    def relaunch_ball(self):
        ball_pos = self.init_qpos[self.ball_posadr: self.ball_posadr + 3]
        ball_vel = self.init_qvel[self.ball_dofadr: self.ball_dofadr + 6] 
        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.sim.model.body_pos[self.id_info.ball_bid] = ball_pos
            self.init_qpos[self.ball_posadr: self.ball_posadr + 3] = ball_pos

        if self.ball_qvel:
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel[:3] = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr: self.ball_dofadr + 3] = ball_vel[:3]
        self.sim.data.qpos[self.ball_posadr: self.ball_posadr + 3] = ball_pos
        self.sim.data.qvel[self.ball_dofadr: self.ball_dofadr + 6] = ball_vel

    def step(self, a, **kwargs):
        # Clip action to our defined action space based on XML ctrlrange
        a = np.clip(a, self.action_space.low, self.action_space.high)
        return super().step(a, **kwargs)


class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.paddle_sid = model.site("paddle").id
        self.paddle_bid = model.body("paddle").id
        self.ball_sid = model.site("pingpong").id
        self.ball_bid = model.body("pingpong").id
        self.ball_gid = model.geom("pingpong").id
        self.own_half_gid = model.geom("coll_own_half").id
        self.paddle_gid = model.geom("pad").id
        self.opponent_half_gid = model.geom("coll_opponent_half").id
        self.ground_gid = model.geom("ground").id
        self.net_gid = model.geom("coll_net").id


class PingpongContactLabels(enum.Enum):
    PADDLE = 0
    OWN = 1
    OPPONENT = 2
    GROUND = 3
    NET = 4
    ENV = 5


class ContactTrajIssue(enum.Enum):
    OWN_HALF = 0
    MISS = 1
    NO_PADDLE = 2
    DOUBLE_TOUCH = 3


def get_ball_contact_labels(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom2, id_info)
        elif model.geom(con.geom2).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom1, id_info)


def geom_id_to_label(geom_id, id_info: IdInfo):
    if geom_id == id_info.paddle_gid:
        return PingpongContactLabels.PADDLE
    elif geom_id == id_info.own_half_gid:
        return PingpongContactLabels.OWN
    elif geom_id == id_info.opponent_half_gid:
        return PingpongContactLabels.OPPONENT
    elif geom_id == id_info.net_gid:
        return PingpongContactLabels.NET
    elif geom_id == id_info.ground_gid:
        return PingpongContactLabels.GROUND
    else:
        return PingpongContactLabels.ENV


def evaluate_pingpong_trajectory(contact_trajectory: List[set]):
    has_hit_paddle = False
    has_bounced_from_paddle = False
    has_bounced_from_table = False
    own_contact_count = 0
    own_contact_phase_done = False

    for s in contact_trajectory:
        if PingpongContactLabels.PADDLE not in s and has_hit_paddle:
            has_bounced_from_paddle = True
        if PingpongContactLabels.PADDLE in s and has_bounced_from_paddle:
            return ContactTrajIssue.DOUBLE_TOUCH
        if PingpongContactLabels.PADDLE in s:
            has_hit_paddle = True
        if PingpongContactLabels.OWN in s:
            if not has_bounced_from_table:
                has_bounced_from_table = True
                own_contact_count = 1
            elif not own_contact_phase_done:
                own_contact_count += 1
                if own_contact_count > 2:
                    own_contact_phase_done = True
                    return ContactTrajIssue.OWN_HALF
            else:
                return ContactTrajIssue.OWN_HALF
        elif has_bounced_from_table:
            own_contact_phase_done = True

        if PingpongContactLabels.OPPONENT in s:
            if has_hit_paddle:
                return None
            else:
                return ContactTrajIssue.NO_PADDLE

    return ContactTrajIssue.MISS
