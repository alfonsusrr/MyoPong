# Hierarchical Model Architecture

## Physics-Based High Policy Implementation (`modules/models/hierarchical.py`)

The physics-based high policy is implemented as a **Gymnasium Wrapper** (`HierarchicalTableTennisWrapper`) that guides the agent by augmenting its observation with a physics-derived "goal" and providing alignment rewards.

### 1. Implementation Overview
The core idea is to transform a complex motor control task into a "tracking" task where the policy learns to align the paddle with a pre-calculated ideal state.

- **Observation Augmentation**: It expands the observation space by **+7 features**:
    - **3D Predicted Ball Position**: Where the ball is expected to be when it crosses the paddle's X-plane.
    - **4D Ideal Paddle Orientation**: A quaternion representing the optimal paddle angle to return the ball to the opponent's side.
- **Analytic Physics Engine (`predict_ball_trajectory`)**:
    - Calculates ballistics (parabolic flight) and **table bounces** using MuJoCo constants (gravity, restitution).
    - Determines a return trajectory that clears the net by calculating a "virtual net height."
    - Uses the **angle bisector** between the incoming ball velocity and the outgoing target path to derive the ideal paddle orientation.

### 2. Model Flow
The data flow within the wrapper during a single environment step follows these phases:

1. **State Extraction**: The wrapper pulls raw simulation data (ball position/velocity, paddle position) from the environment's `obs_dict`.
2. **Goal Calculation & Smoothing**:
    - **Prediction**: Every `update_freq` steps, it runs the physics prediction to update the 7D goal.
    - **Freeze Logic**: Once the ball passes a threshold ($X > 1.4m$), the goal is "frozen" to prevent the target from shifting during the critical contact phase.
    - **Slew Rate Limiting**: The goal is smoothed using max-delta constraints for both position and rotation to prevent sudden jumps that would be physically impossible for the agent to follow.
3. **Observation Feedback**: The smoothed 7D goal is concatenated to the original observation vector and passed to the neural network.
4. **Reward Shaping**: The wrapper calculates **Alignment Rewards** (exponentially decaying based on error):
    - `alignment_y` / `alignment_z`: Rewards the paddle for being at the correct height and lateral position.
    - `paddle_quat_goal`: Rewards the paddle for matching the ideal hitting angle.
    - `pelvis_alignment`: Encourages the body to position itself optimally relative to the predicted hit.
5. **Execution**: These rewards are added to the environment's base reward, providing a dense signal that guides the agent toward the physics-based setpoints.

## Physics Model Formulas (Simplified)

### 1. Ball Trajectory Prediction
The time to impact $t_{imp}$ is determined by the distance to the paddle's X-plane:
\[ t_{imp} = \frac{p_x - b_x}{v_x} \]

The predicted position $(y_{pred}, z_{pred})$ at $t_{imp}$ is calculated via ballistic flight:
\[ y_{pred} = y_0 + v_y t_{imp} \]
\[ z_{pred} = z_0 + v_z t_{imp} - \frac{1}{2}g t_{imp}^2 \]

### 2. Table Bounce Logic
If a bounce occurs at $t_{hit} < t_{imp}$, the vertical velocity $v_z$ is reflected by the restitution coefficient $e$:
\[ v_{z, bounce} = -e (v_z - g t_{hit}) \]
The final $z_{pred}$ is then updated for the remaining time $\Delta t = t_{imp} - t_{hit}$:
\[ z_{pred} = z_{table} + v_{z, bounce} \Delta t - \frac{1}{2}g \Delta t^2 \]

### 3. Ideal Paddle Orientation
The optimal normal vector $\mathbf{n}$ is the unit bisector between the incoming ball direction $\mathbf{d}_{in}$ and the desired return direction $\mathbf{d}_{out}$ to the opponent's target $\mathbf{T}$:
\[ \mathbf{d}_{out} = \text{normalize}(\mathbf{T} - \mathbf{P}_{pred}) \]
\[ \mathbf{d}_{in} = \text{normalize}(\mathbf{V}_{pred}) \]
\[ \mathbf{n}_{ideal} = \text{normalize}(\mathbf{d}_{out} - \mathbf{d}_{in}) \]

The paddle orientation $q$ is solved as the shortest arc rotation from the paddle's rest normal $[-1, 0, 0]^T$ to $\mathbf{n}_{ideal}$.

