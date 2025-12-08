from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class TqdmProgressCallback(BaseCallback):
  def __init__(self, total_steps: int, verbose: int = 0):
    super().__init__(verbose)
    self.total_steps = max(0, total_steps)
    self.pbar = None
    self.last_num_timesteps = 0

  def _on_training_start(self) -> None:
    initial = min(self.num_timesteps, self.total_steps)
    self.pbar = tqdm(
      total=self.total_steps,
      initial=initial,
      desc="Training Progress",
      unit="step",
      leave=True,
    )
    self.last_num_timesteps = self.num_timesteps

  def _on_step(self) -> bool:
    if self.pbar is None:
      return True

    delta = self.num_timesteps - self.last_num_timesteps
    if delta > 0:
      self.pbar.update(delta)
      self.last_num_timesteps = self.num_timesteps
    return True

  def _on_training_end(self) -> None:
    if self.pbar is not None:
      self.pbar.close()
