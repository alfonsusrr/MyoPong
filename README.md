# MyoPong: Goal-Conditioned Synergistic Action and Latent Exploration for Musculoskeletal Table Tennis Control with Training-Free High-Level Policy

MyoPong is a framework designed to tackle the MyoChallenge Table-Tennis task using a musculoskeletal model with up to 210 muscles. It combines a training-free, physics-based high-level planner with a low-level PPO actor integrated with muscle synergy extraction and latent exploration. This hierarchical architecture achieves a near-optimal 94% success rate, demonstrating the effectiveness of combining hierarchical structures with inductive biases in mastering musculoskeletal control.

![Combined Demo](samples/selected_samples/demo/combined_samples.mp4)

## Table of Contents

- [Setup](#setup)
  - [Using Conda](#using-conda)
  - [Using UV](#using-uv)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Features](#features)

## Setup

This project supports two package management systems: **Conda** and **UV**. Choose the one that best fits your workflow.

### Using Conda

1. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate MyoPong
   ```

2. **Install additional dependencies (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

The `environment.yml` file will create a Python 3.10 environment with all necessary dependencies.

### Using UV

1. **Install UV** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or if you prefer to use the lock file:
   ```bash
   uv sync
   ```

The `pyproject.toml` and `uv.lock` files contain all dependency specifications for UV-based installation.

## Training

The training script (`train.py`) provides a unified interface for training PPO agents with various configurations.

### Basic Training

Train a basic PPO agent:

```bash
python train.py \
    --env-id myoChallengeTableTennisP1-v0 \
    --difficulty 1 \
    --total-timesteps 20000000 \
    --num-envs 4
```

### Advanced Features

#### Using Synergy Action Reformulation (SARL)

```bash
python train.py \
    --use-sarl \
    --sar-dir SAR/data \
    --phi 0.8 \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

#### Using Hierarchical Policies

```bash
python train.py \
    --use-hierarchical \
    --update-freq 10 \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

#### Using Lattice Exploration

```bash
python train.py \
    --use-lattice \
    --lattice-alpha 1.0 \
    --std-clip-min 1e-3 \
    --std-clip-max 10.0 \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

#### Using LSTM (Recurrent PPO)

```bash
python train.py \
    --use-lstm \
    --lstm-hidden-size 256 \
    --n-lstm-layers 1 \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

### Combined Features

You can combine multiple features (except LSTM and Lattice, which are incompatible):

```bash
python train.py \
    --use-hierarchical \
    --use-sarl \
    --sar-dir SAR/data \
    --use-lattice \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000 \
    --num-envs 4 \
    --batch-size 2048 \
    --learning-rate 1e-4
```

### Key Training Parameters

- `--env-id`: Gymnasium environment ID (default: `myoChallengeTableTennisP1-v0`)
- `--difficulty`: Curriculum difficulty level 0-4 (default: 1)
- `--total-timesteps`: Total training timesteps (default: 20000000)
- `--num-envs`: Number of parallel environments (default: 4)
- `--batch-size`: Training batch size (default: 2048)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--eval-freq`: Evaluation frequency in steps (default: 100000)
- `--checkpoint-freq`: Checkpoint frequency in steps (default: 1000000)
- `--wandb-project`: Weights & Biases project name (optional)
- `--log-dir`: Directory for logs and checkpoints (default: `runs/unified_ppo`)

### Resuming Training

To resume from a checkpoint:

```bash
python train.py \
    --resume-from-checkpoint path/to/checkpoint.zip \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

### Monitoring Training

Training progress can be monitored via:
- **TensorBoard**: Logs are saved to the `--log-dir` directory
- **Weights & Biases**: Use `--wandb-project` to enable W&B logging

## Evaluation

The evaluation script (`eval.py`) allows you to evaluate trained models on the table tennis environment.

### Basic Evaluation

```bash
python eval.py \
    --model-path path/to/model.zip \
    --env-id myoChallengeTableTennisP1-v0 \
    --difficulty 1 \
    --eval-envs 12 \
    --eval-episodes 400
```

### Evaluation with Feature Flags

**Important**: The feature flags (`--use-sarl`, `--use-hierarchical`, `--use-lattice`, `--use-lstm`) must match the configuration used during training.

```bash
# Evaluate a model trained with SARL
python eval.py \
    --model-path path/to/sarl_model.zip \
    --use-sarl \
    --sar-dir SAR/data \
    --phi 0.8 \
    --env-id myoChallengeTableTennisP1-v0 \
    --eval-episodes 400

# Evaluate a hierarchical model
python eval.py \
    --model-path path/to/hierarchical_model.zip \
    --use-hierarchical \
    --update-freq 10 \
    --env-id myoChallengeTableTennisP1-v0 \
    --eval-episodes 400

# Evaluate a model with Lattice exploration
python eval.py \
    --model-path path/to/lattice_model.zip \
    --use-lattice \
    --env-id myoChallengeTableTennisP1-v0 \
    --eval-episodes 400
```

### Evaluation Output

The evaluation script outputs:
- **Success Rate (score)**: Percentage of successful ball returns
- **Effort**: Average effort metric
- **Standard deviations**: Statistical confidence intervals

Example output:
```
========================================
        EVALUATION RESULTS        
========================================
Model: path/to/model.zip
Duration: 123.45s
Difficulty: 1
Episodes: 400
----------------------------------------
score          :   0.8500 +/-   0.0234
effort         :   0.0123 +/-   0.0012
========================================
```

### Rendering Evaluation

To render evaluation episodes (may be slower):

```bash
python eval.py \
    --model-path path/to/model.zip \
    --render \
    --video-dir eval_videos \
    --eval-envs 4 \
    --eval-episodes 10
```

## Project Structure

```
MyoPong/
├── train.py                 # Main training script
├── eval.py                  # Evaluation script
├── sample.py                # Sample trajectory generation
├── modules/                 # Core modules for the project
│   ├── training/           # Training utilities
│   ├── models/             # Model architectures (hierarchical, lattice)
│   ├── envs/              # Environment wrappers and curriculum
│   └── callback/          # Callbacks (evaluation, checkpointing)
├── physics_pong/            # Physics-based pong environment (6-DoF)
│   ├── sample_pong.py      # Sample trajectory generation of 6-DoF environment
│   ├── eval_physics.py     # Evaluation script of 6-DoF environment
│   └── README.md           # Documentation for physics_pong
├── SAR/                    # Synergy Action Reformulation components
│   ├── data/               # Pre-trained SAR artifacts (ica.pkl, pca.pkl, scaler.pkl, etc.)
│   └── README.md           # Documentation for SAR
├── visualization/         # Visualization tools
├── checkpoints/           # Saved model checkpoints
├── runs/                  # Training logs and outputs
├── images/                # Demo videos and images
├── environment.yml        # Conda environment file
├── pyproject.toml         # UV/Python project configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

### Supported Algorithms
- **PPO** (Proximal Policy Optimization)
- **RecurrentPPO** (LSTM-based PPO)

### Advanced Techniques
- **Synergy Action Reformulation (SARL)**: Uses muscle synergies for more natural control
- **Hierarchical Policies**: Goal-conditioned hierarchical reinforcement learning
- **Lattice Exploration**: Advanced exploration strategy for continuous action spaces
- **Curriculum Learning**: Progressive difficulty levels (0-4)

### Environment
- **MyoSuite Table Tennis**: Realistic musculoskeletal simulation
- **Multiple Phases**: P1 and P2 evaluation phases
- **Customizable Rewards**: Standard and small reward configurations

