import numpy as np
from scipy.spatial.transform import Rotation as R

def _safe_unit(v, fallback):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.broadcast_to(fallback, v.shape).copy(), where=n > 1e-9)

def predict_ball_trajectory(ball_pos, ball_vel, paddle_pos, 
                             gravity=9.81, 
                             table_z=0.785, 
                             ball_radius=0.02, 
                             restitution=0.9,
                             net_height=0.95,
                             default_target=np.array([-0.9, 0.0, 0.95])):
    """
    Predict ball position at the paddle X-plane and compute an "ideal" paddle orientation
    following the logic in pong_v0.py (including net clearance and dynamic target Z).
    
    Args:
        ball_pos: Current ball position (3,) or (N, 3)
        ball_vel: Current ball velocity (3,) or (N, 3)
        paddle_pos: Current paddle position (3,) or (N, 3)
        gravity: Gravity magnitude (positive)
        table_z: Z-coordinate of the table surface (top of table + ball radius)
        ball_radius: Radius of the ball
        restitution: Coefficient of restitution for table bounce
        net_height: Height of the net top (table_height + net_half_height)
        default_target: Default target point on opponent side
        
    Returns:
        pred_ball_pos: Predicted ball position at paddle x-plane
        paddle_ori_ideal: Ideal paddle quaternion [w, x, y, z]
    """
    ball_pos = np.array(ball_pos)
    ball_vel = np.array(ball_vel)
    paddle_pos = np.array(paddle_pos)
    
    # --- 1. Analytic Fallback Prediction (from pong_v0.py lines 292-365) ---
    reach_err = paddle_pos - ball_pos
    err_x = reach_err[..., 0]
    vx = ball_vel[..., 0]
    eps_vx = 1e-3

    dt = np.zeros_like(err_x)
    valid = (err_x > 0.0) & (vx > eps_vx)
    dt = np.divide(err_x, vx, out=dt, where=valid)
    dt = np.clip(dt, 0.0, 2.0)

    z_contact = table_z 

    # Ballistic prediction
    x_pred = np.broadcast_to(paddle_pos[..., 0], err_x.shape)
    y0 = ball_pos[..., 1]
    z0 = ball_pos[..., 2]
    vy0 = ball_vel[..., 1]
    vz0 = ball_vel[..., 2]

    # Unbounced
    y_pred = y0 + vy0 * dt
    z_pred = z0 + vz0 * dt - 0.5 * gravity * (dt ** 2)
    vz_pred = vz0 - gravity * dt

    # Table bounce check
    a = -0.5 * gravity
    b = vz0
    c = z0 - z_contact
    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    denom = 2.0 * a
    t_hit = np.divide((-b - sqrt_disc), denom, out=np.full_like(dt, np.inf), where=np.abs(denom) > 1e-12)
    hit_mask = (t_hit > 0.0) & (t_hit < dt)

    if np.any(hit_mask):
        vz_hit = vz0 - gravity * t_hit
        vz_after = -restitution * vz_hit
        dt2 = dt - t_hit
        y_hit = y0 + vy0 * t_hit
        y_pred_b = y_hit + vy0 * dt2
        z_pred_b = z_contact + vz_after * dt2 - 0.5 * gravity * (dt2 ** 2)
        vz_pred_b = vz_after - gravity * dt2

        if y_pred.ndim > 0:
            y_pred[hit_mask] = y_pred_b[hit_mask]
            z_pred[hit_mask] = z_pred_b[hit_mask]
            vz_pred[hit_mask] = vz_pred_b[hit_mask]
        else:
            y_pred = y_pred_b
            z_pred = z_pred_b
            vz_pred = vz_pred_b

    pred_ball_pos = np.stack([x_pred, y_pred, z_pred], axis=-1)
    pred_ball_vel = np.stack([vx, vy0, vz_pred], axis=-1)

    # --- 2. Dynamic target and normal calculation ---
    target_x = default_target[0]
    target_y = default_target[1]
    target_z = default_target[2]

    p_x = pred_ball_pos[..., 0]
    p_z = pred_ball_pos[..., 2]
    
    vx_in = np.abs(pred_ball_vel[..., 0])
    vx_est = np.maximum(vx_in, 0.5)
    
    # Time to net (net at x=0)
    t_to_net = p_x / vx_est
    gravity_drop = 0.5 * gravity * (t_to_net ** 2)
    h_virt_net = net_height + gravity_drop
    
    denom = target_x - p_x
    ratio_net = np.divide(-p_x, denom, out=np.zeros_like(p_x), where=np.abs(denom) > 1e-6)
    
    target_z_required = p_z + np.divide(h_virt_net - p_z, ratio_net, out=np.zeros_like(p_x), where=np.abs(ratio_net) > 1e-6)
    final_target_z = np.clip(np.maximum(target_z, target_z_required), 0.5, 3.0)
    
    if final_target_z.ndim == 0:
         opp_target = np.array([target_x, target_y, float(final_target_z)])
    else:
         ones = np.ones_like(final_target_z)
         opp_target = np.stack([target_x * ones, target_y * ones, final_target_z], axis=-1)

    # --- 3. Reflection Law and Orientation (Scipy implementation) ---
    d_out = opp_target - pred_ball_pos
    d_out = _safe_unit(d_out, np.array([-1.0, 0.0, 0.0]))
    d_in = _safe_unit(pred_ball_vel, np.array([1.0, 0.0, 0.0]))

    n_ideal = _safe_unit(d_out - d_in, np.array([-1.0, 0.0, 0.0]))
    
    # Paddle normal in local frame is [-1, 0, 0]
    a_u = np.array([-1.0, 0.0, 0.0])
    
    # Find shortest arc rotation from a_u to n_ideal using Scipy
    # We'll handle this manually but using Scipy's Rotation object for the final quat
    if n_ideal.ndim == 1:
        n_ideal_batch = n_ideal[None, :]
    else:
        n_ideal_batch = n_ideal
        
    # Cross product for axis
    axes = np.cross(a_u, n_ideal_batch)
    axes_norm = np.linalg.norm(axes, axis=-1, keepdims=True)
    
    # Avoid divide by zero for parallel vectors
    safe_axes = np.divide(axes, axes_norm, out=np.zeros_like(axes), where=axes_norm > 1e-9)
    # If axes_norm is small, dot(a_u, n_ideal) is near 1 or -1
    dots = np.sum(a_u * n_ideal_batch, axis=-1, keepdims=True)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    
    # Rotvec is axis * angle
    rotvecs = safe_axes * angles
    
    # Handle opposite case (dot ~= -1)
    opposite = (dots < -0.999999).squeeze(-1)
    if np.any(opposite):
        # 180 deg around Z
        rotvecs[opposite] = np.array([0.0, 0.0, np.pi])
        
    rots = R.from_rotvec(rotvecs)
    # Scipy is [x, y, z, w], MuJoCo is [w, x, y, z]
    quats = rots.as_quat()
    quats_mujoco = np.stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]], axis=-1)
    
    if n_ideal.ndim == 1:
        paddle_ori_ideal = quats_mujoco[0]
    else:
        paddle_ori_ideal = quats_mujoco

    return pred_ball_pos, paddle_ori_ideal
