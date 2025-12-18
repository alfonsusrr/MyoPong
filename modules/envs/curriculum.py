from typing import Dict, Any


def tabletennis_curriculum_kwargs(difficulty: int = 0) -> Dict[str, Any]:
  """
  Returns keyword arguments for MyoSuite table tennis environments
  implementing the shared curriculum levels used by PPO scripts.
  """

  # -------------------------------------------------------------------------
  # REWARD CONFIGURATIONS
  # -------------------------------------------------------------------------

  # 1. Warmup Rewards (New)
  # Lowers 'act_reg' (effort penalty) to 0.1.
  # Essential for early training so the agent doesn't collapse to save energy
  # before it knows how to hit the ball.
#   warmup_rewards = {
#       "reach_dist": 1,
#       "palm_dist": 1,
#       "paddle_quat": 2,
#       "act_reg": 0.1,  # <--- SIGNIFICANTLY LOWERED (Default is 0.5)
#       "planner_reach_dist": 2.0,  # <--- ADD THIS LINE
#       "ref_pose": 5.0,
#       "torso_up": 2,
#       "sparse": 100,
#       "solved": 1000,
#       "done": -10,
#   }

#   warmup_rewards = {
#       # OLD: "planner_reach_dist": 5.0, "paddle_quat": 2.0

#       # NEW: Balance them.
#       "reach_dist": 1,
#       "palm_dist": 1,
#       "paddle_quat": 4.0,         # INCREASE: Keep the racket face steady!
#       "act_reg": 0.1,
#       "planner_reach_dist": 2.0,  # Lower this so it doesn't over-optimize position
#       "ref_pose": 1.0,            # Keep low to allow movement
#       "torso_up": 2,
#       "sparse": 200,              # Double this to celebrate ANY contact
#       "solved": 1000,
#       "done": -10,
#   }
  
  warmup_rewards = {
      # after 40M

      # NEW: Balance them.
      "reach_dist": 1,
      "palm_dist": 1,
      "paddle_quat": 4.0,         # INCREASE: Keep the racket face steady!
      "act_reg": 0.05,            # LOWERED from 0.1. Hitting hard costs energy; make it cheap.
      "planner_reach_dist": 2.0,  # Lower this so it doesn't over-optimize position
      "ref_pose": 0.5,            # Lowered further. Hitting hard requires breaking stance!
      "swing_vel": 2.0,           # <--- NEW KEY (Incentivize power)
      "torso_up": 2,
      "sparse": 200,              # Double this to celebrate ANY contact
      "solved": 1000,
      "done": -10,
  }

  # 2. Finetuning Rewards (Original)
  # Sparse focus for robustness once the agent can hit consistently.
  finetuning_rewards = {
      "reach_dist": 0.6,
      "palm_dist": 0.6,
      "paddle_quat": 1.5,
      "act_reg": 0.5,
      "torso_up": 2.0,
      "planner_reach_dist": 2.0,  # <--- ADD THIS LINE
      "sparse": 50,
      "solved": 1200,
      "done": -10,
  }

  # -------------------------------------------------------------------------
  # PHYSICS CONSTANTS
  # -------------------------------------------------------------------------

  # Phase 1 Source Pos (Opponent Side): [-1.25, -0.50, 1.40] to [-1.20, -0.45, 1.50]
  p1_low = [-1.25, -0.50, 1.40]
  p1_high = [-1.20, -0.45, 1.50]

  # Phase 2 Source Pos (Wider Opponent Side)
  p2_low = [-1.25, -0.50, 1.40]
  p2_high = [-0.5, 0.50, 1.50]

  # Friction Nominal: [1.0, 0.005, 0.0001]
  # Variations: +/- [0.1, 0.001, 0.00002]
  fric_nom = [1.0, 0.005, 0.0001]
  fric_delta = [0.1, 0.001, 0.00002]

  fric_low = [n - d for n, d in zip(fric_nom, fric_delta)]
  fric_high = [n + d for n, d in zip(fric_nom, fric_delta)]

  # -------------------------------------------------------------------------
  # CURRICULUM LEVELS
  # -------------------------------------------------------------------------

  curriculum_levels = {
      # Level 0: "The Tee Ball"
      # - Ball starts in P1 zone (noisy)
      # - Ball TARGETS the CENTER of the table (No wild corner shots)
      0: {
          "ball_xyz_range": {
              "low":  [-1.25, -0.485, 1.44],
              "high": [-1.20, -0.465, 1.46]
          },
          # Keep this TRUE. We need the agent to learn physics, not memorization.
          "ball_qvel": True,
          # Serve speed control (keeps target_xyz_range unchanged):
          # - 1.0 = default speed
          # - >1.0 = slower serves (longer flight time, lower horizontal velocity)
          #   "ball_flight_time_scale": 1.5,
          "ball_flight_time_scale": 1.0,
          # Target the strip in the middle of the table (Y = -0.1 to 0.1)
          "target_xyz_range": {
              #   "low":  [0.6, -0.1, 0.785],
              #   "high": [1.2,  0.1, 0.785]
              "low": [0.6, 0.05, 0.785],
              "high": [1.2,  0.15, 0.785]
          },
          "weighted_reward_keys": warmup_rewards,
      },
      # Level 1: "Directional Training"
      # - Widen the target to Y = -0.3 to 0.3 (Mid-range angles)
      1: {
          "ball_xyz_range": {
              "low":  p1_low,
              "high": p1_high
          },
          "ball_qvel": True,
          # Wider target area
          "target_xyz_range": {
              "low":  [0.5, -0.3, 0.785],
              "high": [1.35, 0.3, 0.785]
          },
          "weighted_reward_keys": warmup_rewards,
      },

      # ---------------------------------------------------------------------
      # Level 2: "Standard Phase 1" (Challenge Baseline)
      # Goal: Handle standard serves from the opponent.
      # Note: Reverts to default rewards (higher effort penalty).
      # ---------------------------------------------------------------------
      2: {
          "ball_xyz_range": {"low": p1_low, "high": p1_high},
          "ball_qvel": True,  # Use internal solver to target paddle
      },

      # ---------------------------------------------------------------------
      # Level 3: "Phase 2 Coverage" (Wider Angles)
      # Goal: Reach for balls across the full width of the table.
      # ---------------------------------------------------------------------
      3: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
      },

      # ---------------------------------------------------------------------
      # Level 4: "Robustness" (Mass Variations)
      # Goal: Handle paddle mass changes (100g - 150g)
      # ---------------------------------------------------------------------
      4: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
          "paddle_mass_range": (0.1, 0.15),
          "weighted_reward_keys": finetuning_rewards,
      },

      # ---------------------------------------------------------------------
      # Level 5: "Generalization" (Physics Variations)
      # Goal: Full Phase 2 (Advanced) - Mass + Friction + Noise
      # ---------------------------------------------------------------------
      5: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
          "paddle_mass_range": (0.1, 0.15),
          "ball_friction_range": {"low": fric_low, "high": fric_high},
          "qpos_noise_range": {"low": -0.02, "high": 0.02},
          "weighted_reward_keys": finetuning_rewards,
      },
  }

  return curriculum_levels.get(difficulty, {})
