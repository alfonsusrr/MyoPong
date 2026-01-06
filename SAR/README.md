# Synergy Action Reformulation (SAR)

This directory contains the Synergy Action Reformulation (SAR) implementation and pre-trained artifacts for the table tennis environment. SAR reformulates the action space by extracting muscle synergies from expert demonstrations, enabling more natural and efficient control.

## Overview

SAR uses Independent Component Analysis (ICA) and Principal Component Analysis (PCA) to extract muscle synergies from successful policy rollouts. These synergies capture coordinated muscle activation patterns that are task-general, allowing the policy to learn in a lower-dimensional, more structured action space.

### How SAR Works

1. **Synergy Extraction**: Muscle activations from successful episodes are collected and decomposed using PCA (dimensionality reduction) and ICA (independent component extraction).

2. **Action Reformulation**: The action space is expanded to include both:
   - **Synergy space**: Lower-dimensional space representing coordinated muscle patterns
   - **Original space**: Full action space for task-specific fine-tuning

3. **Blending**: Actions are blended between synergy and original spaces using a weight parameter `phi` (default: 0.8), meaning 80% synergy-based and 20% original actions.

## Pre-trained Artifacts

**We provide pre-trained SAR artifacts** extracted from our trained model (`sar_model.zip`). All the artifacts are inside `data` folder. The following files are included:

- `ica.pkl`: Independent Component Analysis model
- `pca.pkl`: Principal Component Analysis model  
- `scaler.pkl`: MinMaxScaler for normalizing synergy activations
- `sar_model.zip`: The pretrained PPO model used to generate these synergies
- `activations.npy`: Collected muscle activations from successful episodes

These artifacts are ready to use and do not require regeneration. Simply use the `--use-sarl` flag when training or evaluating.

## Usage

### Training with SAR

To train a model using SAR, use the `--use-sarl` flag:

```bash
python train.py \
    --use-sarl \
    --sar-dir SAR/data \
    --phi 0.8 \
    --env-id myoChallengeTableTennisP1-v0 \
    --total-timesteps 20000000
```

**Parameters:**
- `--use-sarl`: Enable Synergy Action Reformulation
- `--sar-dir`: Directory containing SAR artifacts (default: `SAR/data`)
- `--phi`: Blending weight between synergy and original actions (default: 0.8)
  - `phi=1.0`: Pure synergy space
  - `phi=0.0`: Pure original space
  - `phi=0.8`: 80% synergy, 20% original (recommended)

### Evaluation with SAR

When evaluating a model trained with SAR, you must include the same flags:

```bash
python eval.py \
    --model-path path/to/sarl_model.zip \
    --use-sarl \
    --sar-dir SAR/data \
    --phi 0.8 \
    --env-id myoChallengeTableTennisP1-v0 \
    --eval-episodes 400
```

**Important**: The SAR configuration (`--use-sarl`, `--sar-dir`, `--phi`) must match the configuration used during training.

## Generating Your Own SAR Artifacts

If you want to extract synergies from your own trained model, follow these steps:

### Step 1: Collect Muscle Activations

Use `collect_synergy.py` to collect muscle activations from a trained model:

```bash
python SAR/collect_synergy.py \
    --checkpoint path/to/your_model.zip \
    --env-id myoChallengeTableTennisP1-v0 \
    --episodes 2000 \
    --percentile 80 \
    --output activations.npy \
    --num-envs 8
```

**Parameters:**
- `--checkpoint`: Path to trained PPO model checkpoint
- `--env-id`: Environment ID (must match training environment)
- `--episodes`: Number of episodes to collect (default: 2000)
- `--percentile`: Reward percentile threshold - only collect activations from episodes above this percentile (default: 80)
- `--output`: Output file for activations (default: `activations.npy`)
- `--deterministic`: Use deterministic policy (optional)
- `--num-envs`: Number of parallel environments (default: 8)

This script will:
1. Load the trained model
2. Run episodes and collect muscle activations from successful episodes (above the percentile threshold)
3. Save activations to a `.npy` file

### Step 2: Compute SAR Artifacts

Use `icapca.py` to compute the ICA, PCA, and scaler from collected activations:

```python
# Edit icapca.py to set:
# - n_syn: Number of synergies (default: 120)
# - activations.npy: Path to your activations file

python SAR/icapca.py
```

Or modify `icapca.py` directly:

```python
import numpy as np
from SAR.icapca import compute_SAR
import joblib

# Load your activations
activations = np.load("activations.npy")

# Compute SAR with desired number of synergies
ica, pca, scaler = compute_SAR(activations, n_syn=120, save=True)

# This will save:
# - ica.pkl
# - pca.pkl  
# - scaler.pkl
```

**Parameters:**
- `n_syn`: Number of synergies to extract (default: 120)
  - More synergies = more expressive but higher dimensional
  - Fewer synergies = more constrained but lower dimensional

### Step 3: Visualize Variance Accounted For (Optional)

Use `plot_vaf.py` to visualize how many synergies are needed to account for variance in the activations:

```bash
python SAR/plot_vaf.py
```

This helps determine the optimal number of synergies (`n_syn`) for your task.

## Files

- `SynergyWrapper.py`: Implementation of `SynNoSynWrapper` that blends synergy and original actions
- `collect_synergy.py`: Script to collect muscle activations from trained models
- `icapca.py`: Script to compute ICA, PCA, and scaler from activations
- `plot_vaf.py`: Script to visualize variance accounted for by different numbers of synergies
- `ica.pkl`: Pre-trained ICA model (provided)
- `pca.pkl`: Pre-trained PCA model (provided)
- `scaler.pkl`: Pre-trained scaler (provided)
- `sar_model.zip`: Pretrained model used to generate the provided artifacts
- `activations.npy`: Collected activations from the pretrained model
- `vaf_by_n_synergies.png`: Visualization of variance accounted for

## Technical Details

### Action Space Reformulation

When SAR is enabled, the action space is expanded:
- **Original action space**: `env.action_space.shape[0]` dimensions
- **Synergy action space**: `pca.components_.shape[0]` dimensions (typically 120)
- **Combined action space**: `synergy_dim + original_dim` dimensions

The `SynNoSynWrapper`:
1. Takes actions in the combined space
2. Transforms synergy actions back to original space via: `PCA⁻¹(ICA⁻¹(Scaler⁻¹(syn_action)))`
3. Blends synergy and original actions: `phi * syn_action + (1 - phi) * original_action`
4. Handles cases where synergy space is smaller than action space (e.g., when actions include non-muscle actuators)

### Why SAR Helps

1. **Structured Exploration**: Synergies provide a natural structure for exploration
2. **Sample Efficiency**: Lower-dimensional synergy space can be learned more efficiently
3. **Biological Plausibility**: Muscle synergies are observed in biological motor control
4. **Transfer Learning**: Synergies learned from one task can potentially transfer to related tasks

## References

SAR is based on the concept of muscle synergies from motor control literature. The implementation uses:
- **PCA**: For dimensionality reduction
- **ICA**: For extracting independent components (synergies)
- **MinMaxScaler**: For normalizing synergy activations to [-1, 1]

