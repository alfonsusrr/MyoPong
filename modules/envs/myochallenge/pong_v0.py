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

from myosuite.utils.quat_math import euler2quat, rotVecQuat
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

    DEFAULT_OBS_KEYS = ['ball_pos', 'ball_vel', 'paddle_pos', "paddle_vel", 'paddle_ori', 'reach_err' , "touching_info", "pred_ball_pos", "paddle_ori_ideal"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": 0,
        "alignment_y": 5,
        "alignment_z": 5,
        "paddle_quat": 5,
        "move_reg": 0.1,
        "hit_stability": 5.0,
        "act_reg": 1.0,
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
            frame_skip: int = 5,
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            ball_xyz_range = None,
            ball_qvel = None,
            ball_flight_time_scale: float = 1.0,
            ball_friction_range = None,
            paddle_mass_range = None,
            target_xyz_range = None,
            rally_count = 1,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.ball_xyz_range = ball_xyz_range
        self.ball_qvel = ball_qvel
        self.ball_flight_time_scale = float(ball_flight_time_scale) if ball_flight_time_scale is not None else 1.0
        self.qpos_noise_range = qpos_noise_range
        self.paddle_mass_range = paddle_mass_range
        self.ball_friction_range = ball_friction_range
        self.target_xyz_range = target_xyz_range

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
                    normalize_act=True,
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

    def calculate_prediction(self, ball_pos, ball_vel, paddle_pos, paddle_vel=None):
        """
        Predict ball position at the paddle X-plane and compute an "ideal" paddle orientation
        that would reflect the ball towards a fixed target.
        
        For single-env usage (1D `ball_pos` / `ball_vel`), we prefer a short MuJoCo forward
        rollout from the current simulator state. This matches the true dynamics specified
        in `myoarm_pong.xml` (drag via `fluidcoef`, contact with table/net, etc.).

        For batched inputs, we fall back to a lightweight analytic approximation.
        """
        # --- helpers (batch-safe) ---
        def _safe_unit(v, fallback):
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            return np.divide(v, n, out=np.broadcast_to(fallback, v.shape).copy(), where=n > 1e-9)

        def _quat_from_two_unit_vecs(a_unit, b_unit):
            # Returns quaternion [w, x, y, z] rotating a_unit -> b_unit (shortest arc)
            dot = np.sum(a_unit * b_unit, axis=-1, keepdims=True)  # (..., 1)
            v = np.cross(a_unit, b_unit)  # (..., 3)
            w = 1.0 + dot  # (..., 1)

            # Handle opposite vectors: choose a stable orthogonal axis (here z-axis for our fixed a=[-1,0,0])
            opposite = (dot < -0.999999).squeeze(-1)
            q = np.concatenate([w, v], axis=-1)  # (..., 4)
            q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
            q = np.divide(q, q_norm, out=np.zeros_like(q), where=q_norm > 1e-12)

            if np.any(opposite):
                # 180deg around +Z gives [-1,0,0] -> [1,0,0]
                q_op = np.array([0.0, 0.0, 0.0, 1.0])
                if q.ndim == 1:
                    q = q_op
                else:
                    q = q.copy()
                    q[opposite] = q_op
            return q

        def _sim_predict_ball_at_xplane(paddle_xplane, max_time=2.0):
            """
            Predict ball state when its center reaches the given world X-plane, by stepping
            the MuJoCo simulator forward from the current state and then restoring it.
            Returns (pos(3,), vel(3,), ok(bool)).
            """
            sim = self.sim  # SimScene / DMSimScene wrapper
            model = sim.model
            data = sim.data

            # Save full mutable state we might touch
            qpos0 = data.qpos.copy()
            qvel0 = data.qvel.copy()
            time0 = float(data.time)
            ctrl0 = data.ctrl.copy() if getattr(model, "nu", 0) > 0 else None
            act0 = data.act.copy() if getattr(model, "na", 0) > 0 else None
            
            # Save paddle properties if we are going to modify them
            # We want to essentially remove the paddle from collision or move it out of the way.
            # Easiest is to move the paddle far away in joint space.
            # Paddle joints start at self.paddle_posadr
            
            try:
                # Freeze paddle by holding position actuators at current joint positions
                # OR better: move paddle joints far away so it doesn't hit ball during prediction
                # Paddle is joint "paddle_x" etc.
                # Actually, setting qpos to something huge might break physics stability if constraints are violated.
                # Let's just zero velocity and hope it doesn't move much? 
                # Or set the joint limits to be somewhere else?
                # Safer: disable collision for paddle geom.
                # But conaffinity is in model.
                
                # Let's try zeroing velocities and setting control to hold current position (if possible)
                # But the paddle might be IN THE WAY right now.
                # If we are predicting for future intersection, and the paddle is currently blocking, 
                # the rollout will hit the paddle.
                
                # Best approach: Teleport paddle out of the way (e.g. z = +10.0)
                # "paddle_z" is the 3rd joint of paddle.
                # Range is -0.3 to 0.1 in actuator, but joint range is -2 to 2.
                # Let's set paddle_z joint value to 1.5 (high up)
                
                paddle_z_adr = self.paddle_posadr + 2
                data.qpos[paddle_z_adr] = 1.5 
                
                # Also zero velocities for paddle
                pd = int(self.paddle_dofadr)
                data.qvel[pd:pd+6] = 0.0
                
                # Step until ball crosses x-plane (from smaller x to larger x)
                # Ball moves +X in this env setup.
                
                dt_sim = float(model.opt.timestep)
                n_steps = int(max(1, np.ceil(max_time / max(dt_sim, 1e-9))))

                ball_sid = self.id_info.ball_sid
                prev_pos = data.site_xpos[ball_sid].copy()
                prev_err = paddle_xplane - float(prev_pos[0])

                ok = False
                out_pos = prev_pos.copy()
                out_vel = self.get_sensor_by_name(model, data, "pingpong_vel_sensor").copy()

                # If already passed, return current
                if prev_err <= 0:
                    return prev_pos, out_vel, True

                for _ in range(n_steps):
                    # Advance physics by one internal step (no rendering).
                    # DMSimScene implements advance(substeps, render).
                    # Check if advance exists, else use standard step
                    if hasattr(sim, "advance"):
                        sim.advance(substeps=1, render=False)
                    else:
                        # Fallback for standard mujoco/dm_control sim
                        sim.step()
                        
                    curr_pos = data.site_xpos[ball_sid].copy()
                    curr_err = paddle_xplane - float(curr_pos[0])

                    # Detect crossing: err goes from >0 to <=0
                    if (prev_err > 0.0) and (curr_err <= 0.0):
                        # Linear interpolate between prev and curr for position at plane
                        denom = (prev_err - curr_err)
                        alpha = prev_err / denom if abs(denom) > 1e-12 else 0.0
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        out_pos = prev_pos + alpha * (curr_pos - prev_pos)
                        out_pos[0] = float(paddle_xplane)

                        # Use current sensor velocity as a good approximation at crossing
                        out_vel = self.get_sensor_by_name(model, data, "pingpong_vel_sensor").copy()
                        ok = True
                        break

                    prev_pos = curr_pos
                    prev_err = curr_err
                    
                    # Safety check: if ball falls off table (Z < 0), stop
                    if curr_pos[2] < 0.0:
                        break

                return out_pos, out_vel, ok
            except Exception as e:
                # If anything fails, return None
                return None, None, False
            finally:
                # Restore state and forward
                data.qpos[:] = qpos0
                data.qvel[:] = qvel0
                data.time = time0
                if ctrl0 is not None:
                    data.ctrl[:] = ctrl0
                if act0 is not None:
                    data.act[:] = act0
                
                # Use sim.forward() to ensure consistency with wrapper
                sim.forward()

        # --- core ---
        # If we have single-environment vectors, use MuJoCo rollout for best fidelity.
        # This accounts for air resistance, table friction, and complex bounces.
        use_sim_rollout = False
        if use_sim_rollout:
            paddle_xplane = float(paddle_pos[0])

            # If ball isn't moving towards +X plane, don't predict forward.
            # Or if it's already passed.
            if float(ball_vel[0]) <= 1e-4 or float(paddle_xplane - ball_pos[0]) <= 0.0:
                pred_ball_pos = np.array([paddle_xplane, float(ball_pos[1]), float(ball_pos[2])], dtype=float)
                pred_ball_vel = np.array(ball_vel, dtype=float)
            else:
                # Bound rollout horizon by a coarse time-to-plane estimate.
                approx_dt = float((paddle_xplane - ball_pos[0]) / max(float(ball_vel[0]), 1e-4))
                max_time = float(np.clip(approx_dt * 1.5, 0.05, 2.0))
                
                pred_ball_pos, pred_ball_vel, ok = _sim_predict_ball_at_xplane(paddle_xplane, max_time=max_time)
                if not ok or pred_ball_pos is None:
                    # Fall back to analytic prediction if rollout fails
                    use_sim_rollout = False

        if not use_sim_rollout:
            # --- Batched / Analytic Fallback ---
            # Predictive time to paddle x-plane
            reach_err = paddle_pos - ball_pos
            err_x = reach_err[..., 0]
            vx = ball_vel[..., 0]
            eps_vx = 1e-3

            dt = np.zeros_like(err_x)
            valid = (err_x > 0.0) & (vx > eps_vx)
            dt = np.divide(err_x, vx, out=dt, where=valid)
            dt = np.clip(dt, 0.0, 2.0)

            # Gravity magnitude (MuJoCo uses negative Z gravity)
            g = float(-self.sim.model.opt.gravity[2]) if hasattr(self, "sim") else 9.81
            if not np.isfinite(g) or g <= 0:
                g = 9.81

            # Table top plane (use OWN half: where we expect the bounce before paddle hit)
            own_gid = self.id_info.own_half_gid
            table_z = float(self.sim.data.geom_xpos[own_gid][2] + self.sim.model.geom_size[own_gid][2])
            ball_r = float(self.sim.model.geom_size[self.id_info.ball_gid][0])
            z_contact = table_z + ball_r

            # Ballistic prediction (with at most one bounce on the table plane)
            x_pred = np.broadcast_to(paddle_pos[..., 0], err_x.shape)
            y0 = ball_pos[..., 1]
            z0 = ball_pos[..., 2]
            vy0 = ball_vel[..., 1]
            vz0 = ball_vel[..., 2]

            # Unbounced
            y_pred = y0 + vy0 * dt
            z_pred = z0 + vz0 * dt - 0.5 * g * (dt ** 2)
            vz_pred = vz0 - g * dt

            # Check if we'd hit the table before reaching the paddle x-plane:
            # Solve z0 + vz0*t - 0.5*g*t^2 = z_contact for smallest positive t.
            a = -0.5 * g
            b = vz0
            c = z0 - z_contact
            disc = b * b - 4.0 * a * c
            disc = np.maximum(disc, 0.0)
            sqrt_disc = np.sqrt(disc)

            # With a<0, the earlier root is (-b - sqrt_disc)/(2a) if it is positive
            denom = 2.0 * a
            t_hit = np.divide((-b - sqrt_disc), denom, out=np.full_like(dt, np.inf), where=np.abs(denom) > 1e-12)
            hit_mask = (t_hit > 0.0) & (t_hit < dt)

            if np.any(hit_mask):
                # Velocity at table impact
                vz_hit = vz0 - g * t_hit

                # Simple restitution (can be tuned via env attr if you want)
                e = float(getattr(self, "predict_restitution", 0.9))
                e = float(np.clip(e, 0.0, 1.0))

                vz_after = -e * vz_hit
                dt2 = dt - t_hit

                # Propagate after bounce from (y_hit,z_contact) with updated vz
                y_hit = y0 + vy0 * t_hit
                y_pred_b = y_hit + vy0 * dt2
                z_pred_b = z_contact + vz_after * dt2 - 0.5 * g * (dt2 ** 2)
                vz_pred_b = vz_after - g * dt2

                y_pred = np.where(hit_mask, y_pred_b, y_pred)
                z_pred = np.where(hit_mask, z_pred_b, z_pred)
                vz_pred = np.where(hit_mask, vz_pred_b, vz_pred)

            pred_ball_pos = np.stack([x_pred, y_pred, z_pred], axis=-1)

            # Predicted ball velocity at impact (account for gravity + bounce)
            pred_ball_vel = np.stack([vx, vy0, vz_pred], axis=-1)

        # --- Dynamic target and normal calculation (Lob/Ballistic) ---
        # Target a point deep in the opponent's court (high arc, safe landing)
        target_x = -1.5
        target_y = 0.0
        target_z = 1.0 # Default "safe" height

        # --- Dynamic trajectory adjustment (Lob/Ballistic) ---
        # Net top = table_height (0.795) + net_half_height (0.1525) = 0.9475
        net_height = 0.95 
        safe_margin = 0.05

        # 1. Estimate time to net
        p_x = pred_ball_pos[..., 0]
        p_z = pred_ball_pos[..., 2]
        
        # Estimate outgoing x-velocity. Assume elastic reflection roughly preserves |vx|.
        # Incoming vx is positive (towards paddle). Outgoing will be negative (towards net).
        # We use the magnitude of the incoming velocity as a proxy for the return velocity.
        vx_in = np.abs(pred_ball_vel[..., 0])
        vx_est = np.maximum(vx_in, 0.1) # Avoid divide by zero, assume at least 0.1m/s
        
        # Time for ball to travel from paddle (p_x) to net (0)
        # p_x is positive (~1.5m), net is 0.
        t_to_net = p_x / vx_est
        
        # 2. Calculate gravity drop at the net
        g = 9.81
        # z_drop = 0.5 * g * t^2
        gravity_drop = 0.5 * g * (t_to_net ** 2)
        
        # 3. Determine required "linear target height" at the net
        # The linear path must pass through this height at x=0 so that after gravity drops it,
        # it is still above (net_height + margin).
        h_virt_net = net_height + safe_margin + gravity_drop
        
        # 4. Calculate the Target Z required to pass through h_virt_net at x=0
        # Linear interpolation: z(x) = p_z + alpha * (target_z - p_z)
        # alpha at net = (0 - p_x) / (target_x - p_x)
        denom = target_x - p_x
        # Ratio of distance to net vs distance to target
        ratio_net = np.divide(-p_x, denom, out=np.zeros_like(p_x), where=np.abs(denom) > 1e-6)
        
        # Constraint: p_z + ratio_net * (target_z - p_z) >= h_virt_net
        # ratio_net * (target_z - p_z) >= h_virt_net - p_z
        # target_z - p_z >= (h_virt_net - p_z) / ratio_net
        # target_z >= p_z + (h_virt_net - p_z) / ratio_net
        
        target_z_required = p_z + np.divide(h_virt_net - p_z, ratio_net, out=np.zeros_like(p_x), where=np.abs(ratio_net) > 1e-6)
        
        # Use the higher of default or calculated required z
        # Also clamp to reasonable max to prevent shooting at the ceiling (e.g. 4.0m)
        final_target_z = np.clip(np.maximum(target_z, target_z_required), 0.0, 4.0)
        
        if final_target_z.ndim == 0:
             opp_target = np.array([target_x, target_y, float(final_target_z)])
        else:
             # Handle batch case
             ones = np.ones_like(final_target_z)
             opp_target = np.stack([target_x * ones, target_y * ones, final_target_z], axis=-1)

        d_out = opp_target - pred_ball_pos
        d_out = _safe_unit(d_out, np.array([-1.0, 0.0, 0.0]))

        # Relative velocity logic for moving paddle
        if paddle_vel is None:
            # Handle both single and batch cases for default
            if pred_ball_vel.ndim == 1:
                paddle_vel = np.zeros(3)
            else:
                paddle_vel = np.zeros_like(pred_ball_vel)
        
        # v_rel_in = ball_vel - paddle_vel (relative velocity at impact)
        v_rel_in = pred_ball_vel - paddle_vel
        v_rel_in_mag = np.linalg.norm(v_rel_in, axis=-1, keepdims=True)
        
        # We want v_ball_out = beta * d_out
        # Energy conservation: |v_ball_out - paddle_vel| = |v_rel_in|
        # |beta * d_out - paddle_vel|^2 = v_rel_in_mag^2
        # beta^2 - 2*beta*(d_out . paddle_vel) + |paddle_vel|^2 - v_rel_in_mag^2 = 0
        
        d_out_dot_vp = np.sum(d_out * paddle_vel, axis=-1, keepdims=True)
        vp_mag_sq = np.sum(paddle_vel**2, axis=-1, keepdims=True)
        
        # Quadratic: beta^2 + b*beta + c = 0
        # b = -2 * d_out_dot_vp
        # c = vp_mag_sq - v_rel_in_mag^2
        
        discriminant = d_out_dot_vp**2 - vp_mag_sq + v_rel_in_mag**2
        discriminant = np.maximum(discriminant, 0.0)
        
        # Speed of ball after impact in world frame (positive root)
        beta = d_out_dot_vp + np.sqrt(discriminant)
        
        v_ball_out = beta * d_out
        
        # Normal is parallel to change in velocity (Impulse direction)
        n_ideal = _safe_unit(v_ball_out - pred_ball_vel, np.array([-1.0, 0.0, 0.0]))

        # Ensure normal points roughly towards -X (paddle facing direction in this model)
        flip = (n_ideal[..., 0] > 0.0)
        if np.any(flip):
            n_ideal = n_ideal.copy()
            if n_ideal.ndim == 1:
                n_ideal *= -1.0
            else:
                n_ideal[flip] *= -1.0

        # Convert n_ideal to quaternion aligning reference normal [-1, 0, 0] to n_ideal
        a_unit = np.array([-1.0, 0.0, 0.0])
        if n_ideal.ndim == 1:
            a_u = a_unit
        else:
            a_u = np.broadcast_to(a_unit, n_ideal.shape)
        a_u = _safe_unit(a_u, np.array([-1.0, 0.0, 0.0]))
        b_u = _safe_unit(n_ideal, np.array([-1.0, 0.0, 0.0]))
        paddle_ori_ideal = _quat_from_two_unit_vecs(a_u, b_u)

        return pred_ball_pos, n_ideal, paddle_ori_ideal

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        # Core paddle and ball observations
        obs_dict["ball_pos"] = sim.data.site_xpos[self.id_info.ball_sid].copy()
        obs_dict["ball_vel"] = self.get_sensor_by_name(sim.model, sim.data, "pingpong_vel_sensor").copy()

        # Use the pad geom position for more accurate alignment/reach
        obs_dict["paddle_pos"] = sim.data.geom_xpos[self.id_info.paddle_gid].copy()
        obs_dict["paddle_vel"] = self.get_sensor_by_name(sim.model, sim.data, "paddle_vel_sensor").copy()
        obs_dict["paddle_ori"] = sim.data.body_xquat[self.id_info.paddle_bid].copy()

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

        # Predictive ball position at a stable impact X-plane
        ball_vel = obs_dict['ball_vel']
        ball_pos = obs_dict['ball_pos']
        paddle_pos = obs_dict['paddle_pos']

        # Use a fixed impact plane for more stable prediction
        # Paddle x range is [1.5, 1.8] (base 1.8 + ctrl [-0.3, 0])
        stable_paddle_pos = paddle_pos.copy()
        stable_paddle_pos[0] = 1.62 

        pred_ball_pos, n_ideal, paddle_ori_ideal = self.calculate_prediction(ball_pos, ball_vel, stable_paddle_pos, obs_dict.get('paddle_vel'))
        
        obs_dict['pred_ball_pos'] = pred_ball_pos
        obs_dict['n_ideal'] = n_ideal
        obs_dict['paddle_ori_ideal'] = paddle_ori_ideal

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
        reach_err = obs_dict['reach_err']
        reach_dist = np.linalg.norm(reach_err, axis=-1)
        err_x = reach_err[..., 0] 

        # Predicted alignment errors
        pred_err_y = np.abs(obs_dict['paddle_pos'][..., 1] - obs_dict['pred_ball_pos'][..., 1])
        pred_err_z = np.abs(obs_dict['paddle_pos'][..., 2] - obs_dict['pred_ball_pos'][..., 2])
        
        # Closer ball = higher weight for YZ alignment
        ball_closeness_weight = 1.0 / (1.0 + 2 * np.abs(err_x))
        
        # Mask for ball passing paddle (no reward after ball passes paddle)
        active_mask = (err_x > -0.05).astype(float)
        
        # Check if ball has hit our table
        has_bounced = any(PingpongContactLabels.OWN in s for s in self.contact_trajectory)
        
        # Check if ball has already hit the paddle
        has_hit_paddle = any(PingpongContactLabels.PADDLE in s for s in self.contact_trajectory)
        
        # Combined active mask: ball is in front of paddle AND has not been hit yet
        active_alignment_mask = active_mask * (1.0 - float(has_hit_paddle))

        # Center-based alignment
        alignment_y = active_alignment_mask * ball_closeness_weight * np.exp(-3.0 * pred_err_y)
        alignment_z = active_alignment_mask * ball_closeness_weight * np.exp(-5.0 * pred_err_z) if has_bounced else 0.0

        # Orientation error
        n_ideal = obs_dict['n_ideal']
        paddle_ori = obs_dict["paddle_ori"]
        if paddle_ori.ndim == 1:
            n_curr = rotVecQuat(np.array([-1.0, 0.0, 0.0]), paddle_ori)
        else:
            n_curr = np.array([rotVecQuat(np.array([-1.0, 0.0, 0.0]), q.flatten()) for q in paddle_ori.reshape(-1, 4)])
            if paddle_ori.ndim == 2 and paddle_ori.shape[0] == 1:
                n_curr = n_curr[0]
        
        paddle_ori_err = n_curr - n_ideal.squeeze()
        paddle_quat_err = np.linalg.norm(paddle_ori_err, axis=-1)
        paddle_quat_reward = active_alignment_mask * ball_closeness_weight * np.exp(-5.0 * paddle_quat_err) if has_bounced else 0.0

        # Stability rewards
        paddle_vel = np.linalg.norm(obs_dict['paddle_vel'], axis=-1)

        # Angular velocity from sim data 
        paddle_ang_vel_raw = self.sim.data.qvel[self.paddle_dofadr+3 : self.paddle_dofadr+6]
        paddle_ang_vel = np.linalg.norm(paddle_ang_vel_raw)

        # Smooth movement regularizer: quadratic for small jitters, logarithmic for large movements.
        move_reg = -1.0 * np.log(1.0 + paddle_vel + 2.0 * paddle_ang_vel)
        
        # Stability reward during hit (very localized around the ball-paddle contact)
        # Only active for a very small window before and after hitting the ball
        hit_proximity = np.exp(-50.0 * reach_dist)
        hit_stability = active_mask * hit_proximity * np.exp(-paddle_vel - paddle_ang_vel)
        
        # Action Regularization (energy usage)
        # Using control input if available
        act_mag = np.linalg.norm(self.sim.data.ctrl, axis=-1) if self.sim.model.na > 0 else 0.0

        ball_pos = obs_dict["ball_pos"]
        solved = evaluate_pingpong_trajectory(self.contact_trajectory) == None
        paddle_touch = obs_dict['touching_info'][..., 0]

        rwd_dict = collections.OrderedDict((
            ('reach_dist', active_mask * np.exp(-1. * reach_dist)),
            ('alignment_y', alignment_y),
            ('alignment_z', alignment_z),
            ('paddle_quat', paddle_quat_reward),
            ('hit_stability', hit_stability),
            ('move_reg', move_reg), 
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
        hit_paddle = np.mean([np.sum(p['env_infos']['rwd_dict']['sparse']) for p in paths])

        return {
            'score': score,
            'effort':effort,
            'hit_paddle':hit_paddle
        }
    
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
            if isinstance(self.ball_qvel, dict):
                v_low = self.ball_qvel.get("low", [-0.1] * 3)
                v_high = self.ball_qvel.get("high", [0.1] * 3)
                ball_vel = self.np_random.uniform(low=v_low, high=v_high)
            else:
                v_bounds = self.cal_ball_qvel(ball_pos)
                v_low, v_high = v_bounds[1], v_bounds[0]
                ball_vel = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = ball_vel
        
        # Call MujocoEnv.reset directly to avoid BaseV0 muscle logic
        obs = super().reset(reset_qpos=reset_qpos_local, reset_qvel=self.init_qvel,**kwargs)
        self.cur_rally = 0
        return obs

    def cal_ball_qvel(self, ball_qpos):
        if self.target_xyz_range is not None:
            table_upper = self.target_xyz_range["high"]
            table_lower = self.target_xyz_range["low"]
        else:
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

        time_scale = float(self.ball_flight_time_scale) if hasattr(self, "ball_flight_time_scale") else 1.0
        if time_scale != 1.0:
            time_scale = max(1e-6, time_scale)
            t_scaled = t * time_scale
            v_z = (table_upper[2] - ball_qpos[2] - a * (t_scaled**2)) / t_scaled
            t = t_scaled

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
            if isinstance(self.ball_qvel, dict):
                v_low = self.ball_qvel.get("low", [-0.1] * 3)
                v_high = self.ball_qvel.get("high", [0.1] * 3)
                ball_vel[:3] = self.np_random.uniform(low=v_low, high=v_high)
            else:
                v_bounds = self.cal_ball_qvel(ball_pos)
                v_low, v_high = v_bounds[1], v_bounds[0]
                ball_vel[:3] = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr: self.ball_dofadr + 3] = ball_vel[:3]
        self.sim.data.qpos[self.ball_posadr: self.ball_posadr + 3] = ball_pos
        self.sim.data.qvel[self.ball_dofadr: self.ball_dofadr + 6] = ball_vel

    def step(self, a, **kwargs):
        # if self.normalize_act:
        #     # Map [-1, 1] to ctrlrange
        #     a = np.clip(a, -1.0, 1.0)
        #     ctrlrange = self.sim.model.actuator_ctrlrange
        #     a = (a + 1.0) / 2.0 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
        # else:
        #     # Clip action to our defined action space based on XML ctrlrange
        #     a = np.clip(a, self.action_space.low, self.action_space.high)
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
