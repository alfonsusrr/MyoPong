from typing import Dict, Any


def tabletennis_curriculum_kwargs(difficulty: int = 0) -> Dict[str, Any]:
  """
  Returns keyword arguments for MyoSuite table tennis environments
  implementing the shared curriculum levels used by PPO scripts.
  """

#   # Default rewards (Dense shaping for initial learning)
#   default_rewards = {
#       "reach_dist": 1,
#       "palm_dist": 1,
#       "paddle_quat": 2,
#       "act_reg": 0.5,
#       "torso_up": 2,
#       "sparse": 100,
#       "solved": 1000,
#       "done": -10,
#   }

  # Finetuning rewards (Sparse focus for robustness)
  finetuning_rewards = {
      "reach_dist": 0.6,
      "palm_dist": 0.6,
      "paddle_quat": 1.5,
      "act_reg": 0.5,
      "torso_up": 2.0,
      "sparse": 50,
      "solved": 1200,
      "done": -10,
  }

  # --- Constants from Docs ---
  # Phase 1 Pos: [-1.20, -0.45, 1.50] to [-1.25, -0.50, 1.40]
  p1_low = [-1.25, -0.50, 1.40]
  p1_high = [-1.20, -0.45, 1.50]

  # Phase 2 Pos: [-0.5, 0.50, 1.50] to [-1.25, -0.50, 1.40]
  p2_low = [-1.25, -0.50, 1.40]
  p2_high = [-0.5, 0.50, 1.50]

  # Friction Nominal: [1.0, 0.005, 0.0001]
  # Variations: +/- [0.1, 0.001, 0.00002]
  fric_nom = [1.0, 0.005, 0.0001]
  fric_delta = [0.1, 0.001, 0.00002]

  fric_low = [n - d for n, d in zip(fric_nom, fric_delta)]
  fric_high = [n + d for n, d in zip(fric_nom, fric_delta)]

  curriculum_levels = {
      # Level 0: "The Tee Ball"
      # - Ball starts in P1 zone (noisy)
      # - Ball TARGETS the CENTER of the table (No wild corner shots)
      0: {
          "ball_xyz_range": {
              "low":  [-1.235, -0.485, 1.44],
              "high": [-1.215, -0.465, 1.46]
          },
          # Keep this TRUE. We need the agent to learn physics, not memorization.
          "ball_qvel": True,
          # Target the strip in the middle of the table (Y = -0.1 to 0.1)
          "target_xyz_range": {
              "low":  [0.6, -0.1, 0.785],
              "high": [1.2,  0.1, 0.785]
          },
      },
      # Level 1: "Directional Training"
      # - Widen the target to Y = -0.3 to 0.3 (Mid-range angles)
      1: {
          "ball_xyz_range": {
              "low":  [-1.235, -0.50, 1.44],
              "high": [-1.215, -0.45, 1.46]
          },
          "ball_qvel": True,
          # Wider target area
          "target_xyz_range": {
              "low":  [0.5, -0.3, 0.785],
              "high": [1.35, 0.3, 0.785]
          },
      },
      # Level 2: Phase 1 Box
      # - Position: Phase 1 specific box
      # - Velocity: Full
      2: {
          "ball_xyz_range": {"low": p1_low, "high": p1_high},
          "ball_qvel": True,
      },
      # Level 3: Phase 2 Box (Wider)
      # - Position: Phase 2 full width
      # - Velocity: Full
      3: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
      },
      # Level 4: Phase 2 Spatial Expansion + Mass.
      # Phase 2 Spatial Expansion
      # Add Paddle Mass: 100g - 150g (0.1 - 0.15 kg)
      4: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
          "paddle_mass_range": (0.1, 0.15),  # CORRECTED UNITS (kg)
          "weighted_reward_keys": finetuning_rewards,
      },
      # Level 5: Full Phase 2 (Advanced).
      # Full spatial, Full velocity, Full dynamics (Mass + Friction).
      5: {
          "ball_xyz_range": {"low": p2_low, "high": p2_high},
          "ball_qvel": True,
          "paddle_mass_range": (0.1, 0.15),
          "ball_friction_range": {"low": fric_low, "high": fric_high},
          # Keep your noise choice if helpful
          "qpos_noise_range": {"low": -0.02, "high": 0.02},
          "weighted_reward_keys": finetuning_rewards,
      },
  }

  return curriculum_levels.get(difficulty, {})
