from stable_baselines3.common.callbacks import BaseCallback
import wandb


class WandbCallback(BaseCallback):
  """Lightweight callback to push SB3 training and rollout metrics to W&B."""

  def __init__(self, verbose: int = 0):
    super().__init__(verbose)

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
