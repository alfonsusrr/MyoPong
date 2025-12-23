import os
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

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

