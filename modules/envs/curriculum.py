from typing import Dict, Any


def tabletennis_curriculum_kwargs(difficulty: int = 0, reward_type: str = "small") -> Dict[str, Any]:
  """
  Returns keyword arguments for MyoSuite table tennis environments
  implementing the shared curriculum levels used by PPO scripts.
  """

  # -------------------------------------------------------------------------
  # REWARD CONFIGURATIONS
  # -------------------------------------------------------------------------

  # 1. Standard Rewards (Original scale from tabletennis_v0.py)
  rewards = {
      "reach_dist": 1,
      "palm_dist": 1,
      "paddle_quat": 0,
      "act_reg": .5,
      'torso_up': 2,
      "alignment_y": 2.0,
      "alignment_z": 2.0,
      "paddle_quat_goal": 2.0,
      "pelvis_alignment": 0.0,
      "sparse": 100,
      "solved": 1000,
      'done': -10
  }

  # 2. Small Rewards (Current scale from tabletennis_v0.py)
  small_rewards = {
      "reach_dist": 0.1,
      "palm_dist": 0.1,
      "paddle_quat": 0.0,
      "alignment_y": 0.5,
      "alignment_z": 0.5,
      "paddle_quat_goal": 0.5,
      "pelvis_alignment": 0.0,
      "act_reg": 0.1,
      "torso_up": 0.5,
      "sparse": 5.0,
      "solved": 100.0,
      "done": -5.0,
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
          "ball_qvel": False,
          # Serve speed control (keeps target_xyz_range unchanged):
          # - 1.0 = default speed
          # - >1.0 = slower serves (longer flight time, lower horizontal velocity)
        #   "ball_flight_time_scale": 1.5,
        #   # Target the strip in the middle of the table (Y = -0.1 to 0.1)
        #   "target_xyz_range": {
        #       #   "low":  [0.6, -0.1, 0.785],
        #       #   "high": [1.2,  0.1, 0.785]
        #       "low": [0.6, 0.05, 0.785],
        #       "high": [1.2,  0.15, 0.785]
        #   },
        #   "weighted_reward_keys": warmup_rewards,
      },
      # Level 1: "Directional Training"
      # - Widen the target to Y = -0.3 to 0.3 (Mid-range angles)
      1: {
          "ball_xyz_range": {
              "low":  p1_low,
              "high": p1_high
          },
          "ball_qvel": False,
        #   "weighted_reward_keys": warmup_rewards,
      },

      # ---------------------------------------------------------------------
      # Level 2: "Standard Phase 1" (Challenge Baseline)
      # Goal: Handle standard serves from the opponent.
      # Note: Reverts to default rewards (higher effort penalty).
      # ---------------------------------------------------------------------
      2: {
          "ball_xyz_range": {"low": p1_low, "high": p1_high},
          "ball_qvel": True,
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
        #   "weighted_reward_keys": finetuning_rewards,
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
        #   "weighted_reward_keys": finetuning_rewards,
      },
  }

  kwargs = curriculum_levels.get(difficulty, {}).copy()

  # Set reward weights if not already specified in the level
  if "weighted_reward_keys" not in kwargs:
    if reward_type == "small":
      kwargs["weighted_reward_keys"] = small_rewards.copy()
    elif reward_type == "standard":
      kwargs["weighted_reward_keys"] = rewards.copy()

  # Activate pelvis reward for level 2 and above
  if difficulty >= 2:
    if reward_type == "small":
      kwargs["weighted_reward_keys"]["pelvis_alignment"] = 1.0
    elif reward_type == "standard":
      kwargs["weighted_reward_keys"]["pelvis_alignment"] = 4.0

  return kwargs
