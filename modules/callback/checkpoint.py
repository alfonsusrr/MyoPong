import os
import numpy as np
from pathlib import Path
from typing import Any, Optional
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from modules.callback.evaluator import PeriodicEvaluator

def resolve_checkpoint_path(checkpoint_target: str) -> str:
    """
    Resolves a checkpoint path. If it's a directory, finds the latest .zip file.
    If it's a file without .zip, tries adding it.
    """
    target = Path(checkpoint_target).expanduser().resolve()
    if target.is_dir():
        checkpoints = sorted(
            target.glob("*.zip"), key=lambda path: path.stat().st_mtime
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint archives found in {target}")
        return str(checkpoints[-1])

    if target.is_file():
        return str(target)

    if target.suffix != ".zip":
        zipped_candidate = target.with_suffix(".zip")
        if zipped_candidate.is_file():
            return str(zipped_candidate)

    raise FileNotFoundError(f"Checkpoint path {checkpoint_target} does not exist")

class SaveVecNormalizeCallback(CheckpointCallback):
    """
    Callback for saving a model checkpoint and VecNormalize statistics every `save_freq` steps.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_normalize = self.model.get_vec_normalize_env()
            if isinstance(vec_normalize, VecNormalize):
                # Save with timestep in name
                save_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl")
                vec_normalize.save(save_path)
                
                # Also save a 'latest' version in the main log directory (one level above checkpoints/)
                latest_path = os.path.join(os.path.dirname(self.save_path), "vecnormalize.pkl")
                vec_normalize.save(latest_path)
                
                if self.verbose > 0:
                    print(f"Saving VecNormalize statistics to {save_path}")
                    
        return super()._on_step()

class ScoreThresholdSaveCallback(BaseCallback):
    """
    Callback that saves a checkpoint only when the evaluation score 
    increases by more than a certain threshold.
    """
    def __init__(self, evaluator: PeriodicEvaluator, save_path: str, threshold: float = 0.05, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.evaluator = evaluator
        self.save_path = save_path
        self.threshold = threshold
        self.name_prefix = name_prefix
        self.best_score = -np.inf
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # Check if a new evaluation has finished
        if self.evaluator.last_metrics is not None and self.evaluator._last_eval_step > self._last_eval_step:
            self._last_eval_step = self.evaluator._last_eval_step
            current_score = self.evaluator.last_metrics.get("score", 0.0)
            
            if current_score >= self.best_score + self.threshold:
                if self.verbose > 0:
                    print(f"\n[ScoreThresholdSave] Score improved from {self.best_score:.4f} to {current_score:.4f} (threshold {self.threshold}). Saving checkpoint at step {self.num_timesteps}...")
                self.best_score = current_score
                
                # Save model
                model_path = os.path.join(self.save_path, f"{self.name_prefix}_score_{current_score:.4f}_{self.num_timesteps}_steps.zip")
                self.model.save(model_path)
                
                # Save VecNormalize
                vec_normalize = self.model.get_vec_normalize_env()
                if isinstance(vec_normalize, VecNormalize):
                    vec_save_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_score_{current_score:.4f}_{self.num_timesteps}_steps.pkl")
                    vec_normalize.save(vec_save_path)
                    
                    # Also save a 'latest' version in the main log directory
                    latest_path = os.path.join(os.path.dirname(self.save_path), "vecnormalize.pkl")
                    vec_normalize.save(latest_path)
        return True

