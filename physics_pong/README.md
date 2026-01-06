# Physics Pong Environment

This directory contains tools for working with the simplified 6-DoF Pong environment (`myoChallengePongP0-v0`), which uses a simplified physics model compared to the full table tennis environment.

## Setup

To use the Pong environment, you need to configure MyoSuite to use the custom `myoarm_pong.xml` model file.

### Step 1: Copy the XML Model File

Copy the custom pong XML file to the MyoSuite assets directory:

```bash
# Find your MyoSuite installation path
python -c "import myosuite; import os; print(os.path.dirname(myosuite.__file__))"

# Copy the XML file (replace <MYOSUITE_PATH> with the path above)
cp modules/envs/myochallenge/myoarm_pong.xml <MYOSUITE_PATH>/envs/myo/assets/arm/myoarm_pong.xml
```

Alternatively, if you want to use the local file without modifying the MyoSuite installation, you can modify the environment registration in `modules/envs/myochallenge/__init__.py`:

### Step 2: Update Environment Registration

Edit `modules/envs/myochallenge/__init__.py` and modify the `myoChallengePongP0-v0` registration (around line 96-104) to point to the local XML file:

```python
register_env_with_variants(id='myoChallengePongP0-v0',
        entry_point='myosuite.envs.myo.myochallenge.pong_v0:PongEnvV0',
        max_episode_steps=300,
        kwargs={
            'model_path': curr_dir + '/myoarm_pong.xml',  # Changed from '/../assets/arm/myoarm_pong.xml'
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
```

This changes the path from the MyoSuite assets directory to the local `myoarm_pong.xml` file in the `modules/envs/myochallenge/` directory.

## Heuristic Policy

The `eval_physics.py` script implements a heuristic policy for the 6-DoF Pong environment. This policy uses physics-based predictions to control the paddle.

### How the Heuristic Works

The heuristic policy (`heuristic_policy` function) implements the following strategy:

#### 1. **Ball Trajectory Prediction**
   - Uses pre-calculated `pred_ball_pos` (predicted ball position at impact) from the environment observations
   - Uses `paddle_ori_ideal` (ideal paddle orientation) to determine the correct paddle angle

#### 2. **Paddle Positioning**
   - Calculates the desired joint position by:
     - Taking the predicted ball position
     - Subtracting the base position (`[1.8, 0.5, 1.13]`)
     - Accounting for the paddle offset in the body frame (`[-5.5743e-05, 2.0686e-02, 6.6874e-02]`)
     - Rotating the offset by the ideal paddle orientation
   - Normalizes the position to the control range `[-1, 1]`

#### 3. **Paddle Orientation**
   - Converts the ideal quaternion orientation to Euler angles
   - Normalizes the rotation to the control range `[-1, 1]`

#### 4. **Impact Stabilization**
   The policy freezes the target action when:
   - The ball is very close to impact (`abs(ball_pos_x - paddle_pos_x) < 0.05`)
   - The paddle is already touching the ball (`paddle_touch > 0`)
   - The ball is moving away (hit already happened, `ball_vel_x < -0.05`)

   This prevents the paddle from "chasing" the ball during impact, which can cause instability.

#### 5. **Slew Rate Limiting**
   - Limits the maximum change per step to `0.1` in normalized units
   - Prevents rapid oscillations and "spinning" behavior
   - Allows full range travel in approximately 20 steps (~0.2 seconds at 100Hz)

### Observation Space

The heuristic uses the following observation indices:
- `[0:3]`: Ball position (x, y, z)
- `[3:6]`: Ball velocity (x, y, z)
- `[6:9]`: Paddle position (x, y, z)
- `[9:12]`: Paddle velocity (x, y, z)
- `[19]`: Paddle touch indicator
- `[25:28]`: Predicted ball position at impact
- `[28:32]`: Ideal paddle orientation (quaternion)

### Action Space

The policy outputs 6-DoF actions:
- `[0:3]`: Position control (normalized to `[-1, 1]`)
- `[3:6]`: Rotation control (normalized to `[-1, 1]`)

## Usage

### Evaluate the Heuristic Policy

Run the evaluation script to test the heuristic policy:

```bash
cd physics_pong
python eval_physics.py \
    --env-id myoChallengePongP0-v0 \
    --num-episodes 400 \
    --num-envs 12 \
    --difficulty 5 \
    --seed 42
```

### Options

- `--env-id`: Environment ID (default: `myoChallengePongP0-v0`)
- `--num-episodes`: Total number of episodes to evaluate (default: 400)
- `--num-envs`: Number of parallel environments (default: 12)
- `--difficulty`: Curriculum difficulty level (default: 5, note: pong uses difficulty 5 which maps to specific ball ranges)
- `--seed`: Random seed (default: 42)
- `--render-fails`: Render and save videos of failed episodes
- `--video-dir`: Directory to save failed episode videos (default: `failed_episodes`)

### Example: Render Failed Episodes

To visualize episodes where the heuristic fails:

```bash
python eval_physics.py \
    --env-id myoChallengePongP0-v0 \
    --num-episodes 100 \
    --num-envs 4 \
    --render-fails \
    --video-dir failed_episodes
```

### Output Metrics

The evaluation script reports:
- **Average Reward**: Mean episode reward
- **Success Rate**: Percentage of episodes where the ball was successfully returned
- **Paddle Hit Rate**: Percentage of episodes where the paddle made contact with the ball

Example output:
```
==============================
EVALUATION METRICS (Physics Heuristic)
==============================
Total Episodes:   400
Average Reward:   125.34 +/- 12.45
Success Rate:     85.50% +/- 2.34%
Paddle Hit Rate:  92.25% +/- 1.87%
==============================
```

### Generate Sample Trajectories

The `sample_pong.py` script generates sample trajectory videos using the heuristic policy. This is useful for:
- Visualizing the heuristic policy behavior
- Creating demonstration videos
- Understanding the ball-paddle interaction dynamics

#### Usage

Run the script to generate sample trajectories:

```bash
cd physics_pong
python sample_pong.py
```

#### How It Works

The script:
1. **Creates the environment** with ball randomization ranges matching table tennis curriculum difficulty 3:
   - Ball position range: `{'high': [-0.8, 0.5, 1.5], 'low': [-1.25, -0.5, 1.4]}`
   - Enables `ball_qvel=True` for realistic ball velocity initialization

2. **Runs episodes** using the same heuristic policy as `eval_physics.py`:
   - Each episode runs for up to 500 steps or until termination
   - Uses the same impact stabilization and slew rate limiting logic

3. **Renders videos**:
   - Captures frames from camera 3 (overview camera) at 640x480 resolution
   - Saves videos at 30 FPS
   - Videos are saved to the `pong_samples/` directory

4. **Generates multiple samples**:
   - By default, generates 10 sample trajectories
   - Each sample is saved as `sample_{i}_cam3.mp4` where `i` is the sample number

#### Output

The script creates a `pong_samples/` directory containing:
- `sample_1_cam3.mp4` through `sample_10_cam3.mp4`
- Each video shows a complete episode from the overview camera angle

Example output:
```
Starting sample 1/10...
  Sample 1 saved to pong_samples/sample_1_cam3.mp4 (245 steps)
Starting sample 2/10...
  Sample 2 saved to pong_samples/sample_2_cam3.mp4 (198 steps)
...
All samples completed.
```

#### Customization

To modify the script behavior, edit `sample_pong.py`:
- Change `num_samples` (line 93) to generate more or fewer samples
- Modify `ball_xyz_range` (line 81) to adjust ball initialization
- Change `camera_id` (line 123) to use a different camera angle
- Adjust `range(500)` (line 109) to change maximum episode length

## Files

- `eval_physics.py`: Evaluation script with heuristic policy implementation
- `sample_pong.py`: Script to generate sample trajectory videos using the heuristic policy
- `pong_test.py`: Testing script for debugging and visualization
- `pong_env.py`: Interactive MuJoCo viewer for the pong environment
- `README.md`: This file

## Differences from Table Tennis Environment

The Pong environment (`myoChallengePongP0-v0`) is a simplified version of the table tennis environment:

1. **6-DoF Control**: Direct control of paddle position and orientation (no muscle actuation)
2. **Simplified Physics**: Faster simulation, suitable for rapid prototyping
3. **Same Observation Space**: Uses the same observation structure as table tennis for compatibility

This makes it ideal for:
- Testing heuristic policies
- Rapid iteration on control algorithms
- Understanding the core physics of ball-paddle interaction

