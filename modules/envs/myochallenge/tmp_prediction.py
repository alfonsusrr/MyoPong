def calculate_prediction(self, ball_pos, ball_vel, paddle_pos):
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
        # Only for verification; rollout the mujoco simulator for the prediction
        if False:
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
                    # Fall back to current state if we didn't cross the plane in time.
                    # Or maybe use analytic fallback?
                    # Let's just project linearly as a failsafe
                    pred_ball_pos = np.array([paddle_xplane, float(ball_pos[1]), float(ball_pos[2])], dtype=float)
                    pred_ball_vel = np.array(ball_vel, dtype=float)

            # Ideal paddle normal (Reflection law) using predicted impact point + impact velocity
            opp_target = np.array([-0.7, 0.0, 0.8], dtype=float)
            d_out = _safe_unit(opp_target - pred_ball_pos, np.array([-1.0, 0.0, 0.0]))
            d_in = _safe_unit(pred_ball_vel, np.array([1.0, 0.0, 0.0]))
            n_ideal = _safe_unit(d_out - d_in, np.array([-1.0, 0.0, 0.0]))

            # Keep normal generally pointing -X (paddle facing convention)
            if float(n_ideal[0]) > 0.0:
                n_ideal = -n_ideal

            a_u = np.array([-1.0, 0.0, 0.0], dtype=float)
            paddle_ori_ideal = _quat_from_two_unit_vecs(a_u, _safe_unit(n_ideal, np.array([-1.0, 0.0, 0.0])))
            return pred_ball_pos, n_ideal, paddle_ori_ideal

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

        # Ideal paddle normal (Reflection law) using predicted impact point + impact velocity
        opp_target = np.array([-0.7, 0.0, 0.8])
        d_out = opp_target - pred_ball_pos
        d_out = _safe_unit(d_out, np.array([-1.0, 0.0, 0.0]))

        d_in = _safe_unit(pred_ball_vel, np.array([1.0, 0.0, 0.0]))

        n_ideal = _safe_unit(d_out - d_in, np.array([-1.0, 0.0, 0.0]))
        # Ensure normal points roughly towards -X (paddle facing direction in this model)
        flip = (n_ideal[..., 0] > 0.0)
        if np.any(flip):
            n_ideal = n_ideal.copy()
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