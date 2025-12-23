import numpy as np

def predict_ball_trajectory(ball_pos, ball_vel, paddle_pos, 
                             gravity=9.81, 
                             table_z=0.785, 
                             ball_radius=0.02, 
                             restitution=0.9,
                             net_height=0.95,
                             default_target=np.array([-0.9, 0.0, 0.95])):
    """
    Predict ball position at the paddle X-plane and compute an "ideal" paddle orientation.
    Optimized version with manual quaternion math and minimized array allocations.
    
    Args:
        ball_pos: Current ball position (3,) or (N, 3)
        ball_vel: Current ball velocity (3,) or (N, 3)
        paddle_pos: Current paddle position (3,) or (N, 3)
        gravity: Gravity magnitude (positive)
        table_z: Z-coordinate of the table surface (top of table + ball radius)
        ball_radius: Radius of the ball
        restitution: Coefficient of restitution for table bounce
        net_height: Height of the net top
        default_target: Default target point on opponent side
        
    Returns:
        pred_ball_pos: Predicted ball position at paddle x-plane
        paddle_ori_ideal: Ideal paddle quaternion [w, x, y, z]
    """
    ball_pos = np.asanyarray(ball_pos)
    ball_vel = np.asanyarray(ball_vel)
    paddle_pos = np.asanyarray(paddle_pos)
    
    # --- 1. Analytic Fallback Prediction ---
    # Predictive time to paddle x-plane
    px = paddle_pos[..., 0]
    bx = ball_pos[..., 0]
    vx = ball_vel[..., 0]
    
    err_x = px - bx
    dt = np.zeros_like(err_x)
    # Ball moves +X in this env setup, paddle is at larger X.
    valid = (err_x > 0.0) & (vx > 1e-3)
    dt = np.divide(err_x, vx, out=dt, where=valid)
    np.clip(dt, 0.0, 2.0, out=dt)

    # Ballistic prediction
    y0 = ball_pos[..., 1]
    z0 = ball_pos[..., 2]
    vy0 = ball_vel[..., 1]
    vz0 = ball_vel[..., 2]

    # Unbounced
    y_pred = y0 + vy0 * dt
    z_pred = z0 + vz0 * dt - 0.5 * gravity * (dt ** 2)
    vz_pred = vz0 - gravity * dt

    # Table bounce check: Solve z0 + vz0*t - 0.5*g*t^2 = table_z
    # discriminant = b^2 - 4ac = vz0^2 - 4*(-0.5g)*(z0 - table_z) = vz0^2 + 2g(z0 - table_z)
    disc = vz0 * vz0 + 2.0 * gravity * (z0 - table_z)
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    # Smallest positive root for a = -0.5g is (-vz0 - sqrt_disc) / (2 * -0.5g) = (vz0 + sqrt_disc) / g
    t_hit = (vz0 + sqrt_disc) / gravity
    hit_mask = (t_hit > 0.0) & (t_hit < dt)

    if np.any(hit_mask):
        # Local variables to avoid repeated indexing
        if hit_mask.ndim == 0: # scalar case
            vz_h = vz0 - gravity * t_hit
            vz_a = -restitution * vz_h
            dt2 = dt - t_hit
            y_pred = (y0 + vy0 * t_hit) + vy0 * dt2
            z_pred = table_z + vz_a * dt2 - 0.5 * gravity * (dt2 ** 2)
            vz_pred = vz_a - gravity * dt2
        else:
            tm = t_hit[hit_mask]
            dtm = dt[hit_mask]
            vy0m = vy0[hit_mask]
            vz0m = vz0[hit_mask]
            
            vz_h = vz0m - gravity * tm
            vz_a = -restitution * vz_h
            dt2 = dtm - tm
            
            y_pred[hit_mask] = (y0[hit_mask] + vy0m * tm) + vy0m * dt2
            z_pred[hit_mask] = table_z + vz_a * dt2 - 0.5 * gravity * (dt2 ** 2)
            vz_pred[hit_mask] = vz_a - gravity * dt2

    # --- 2. Dynamic target and orientation ---
    vx_est = np.maximum(np.abs(vx), 0.5)
    # Time for ball to travel from paddle (px) to net (0)
    t_to_net = px / vx_est
    gravity_drop = 0.5 * gravity * (t_to_net ** 2)
    h_virt_net = net_height + gravity_drop
    
    target_x = default_target[0]
    denom = target_x - px
    ratio_net = np.divide(-px, denom, out=np.zeros_like(px), where=np.abs(denom) > 1e-6)
    
    # Required target_z to pass net
    target_z_req = z_pred + np.divide(h_virt_net - z_pred, ratio_net, out=np.zeros_like(px), where=np.abs(ratio_net) > 1e-6)
    final_target_z = np.clip(np.maximum(default_target[2], target_z_req), 0.5, 3.0)
    
    # Orientation: Shortest arc from [-1, 0, 0] to n_ideal
    # n_ideal bisects d_out_u and -d_in_u
    # d_out = opp_target - pred_ball_pos
    dx = target_x - px
    dy = default_target[1] - y_pred
    dz = final_target_z - z_pred
    dn = np.sqrt(dx*dx + dy*dy + dz*dz)
    dn = np.maximum(dn, 1e-9)
    dux, duy, duz = dx/dn, dy/dn, dz/dn
    
    # d_in = pred_ball_vel normalized
    din_n = np.sqrt(vx*vx + vy0*vy0 + vz_pred*vz_pred)
    din_n = np.maximum(din_n, 1e-9)
    dinux, dinuy, dinuz = vx/din_n, vy0/din_n, vz_pred/din_n
    
    # n_ideal ~ d_out_u - d_in_u
    nx, ny, nz = dux - dinux, duy - dinuy, duz - dinuz
    nn = np.sqrt(nx*nx + ny*ny + nz*nz)
    mask_n = nn > 1e-9
    nx = np.divide(nx, nn, out=np.full_like(nx, -1.0), where=mask_n)
    ny = np.divide(ny, nn, out=np.zeros_like(ny), where=mask_n)
    nz = np.divide(nz, nn, out=np.zeros_like(nz), where=mask_n)
    
    # Shortcut quat from a=[-1,0,0] to n=[nx, ny, nz]
    # dot = -nx, v = cross(a, n) = [0, nz, -ny], w = 1 + dot = 1 - nx
    qw = 1.0 - nx
    qx = np.zeros_like(nx)
    qy = nz
    qz = -ny
    qn = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    
    mask_q = qn > 1e-12
    qw = np.divide(qw, qn, out=np.zeros_like(qw), where=mask_q)
    qx = np.divide(qx, qn, out=np.zeros_like(qx), where=mask_q)
    qy = np.divide(qy, qn, out=np.zeros_like(qy), where=mask_q)
    qz = np.divide(qz, qn, out=np.zeros_like(qz), where=mask_q)
    
    # Fix opposite case (nx ~ 1 means dot ~ -1)
    opp = nx > 0.999999
    if np.any(opp):
        if qw.ndim == 0:
            qw, qx, qy, qz = 0.0, 0.0, 0.0, 1.0
        else:
            qw[opp], qx[opp], qy[opp], qz[opp] = 0.0, 0.0, 0.0, 1.0
            
    pred_pos = np.stack([px, y_pred, z_pred], axis=-1)
    pred_ori = np.stack([qw, qx, qy, qz], axis=-1)
    
    return pred_pos, pred_ori
