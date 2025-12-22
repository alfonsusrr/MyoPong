from stable_baselines3.common.callbacks import BaseCallback
import wandb


class WandbCallback(BaseCallback):
  """Lightweight callback to push SB3 training and rollout metrics to W&B."""

  def __init__(self, verbose: int = 0):
    super().__init__(verbose)

  def _on_rollout_start(self) -> None:
    # Log training metrics from the previous iteration's training phase
    metrics = {}
    for k in ['value_loss', 'entropy_loss', 'policy_gradient_loss','loss', 'approx_kl', 'explained_variance', 'std']:
      sb3_key = f"train/{k}"
      if sb3_key in self.logger.name_to_value:
        metrics[sb3_key] = self.logger.name_to_value[sb3_key]
    if metrics:
      wandb.log(metrics, step=self.num_timesteps)

  def _on_training_end(self) -> None:
    # Log final training metrics
    self._on_rollout_start()

  def _on_step(self) -> bool:
    # SB3 logger already has many keys; also log raw infos if available
    if len(self.locals.get('infos', [])) > 0:
      info = self.locals['infos'][0]
      if isinstance(info, dict):
        # Selectively log some useful info keys if present
        for k in ['episode', 'time', 'success', 'reward']:
          if k in info:
            wandb.log({f"env/{k}": info[k]}, step=self.num_timesteps)
    return True
