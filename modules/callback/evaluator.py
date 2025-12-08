import time
from typing import Dict, List, Any, Optional
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor


def evaluate_policy(model: Any, eval_vec: VecMonitor, episodes: int, metrics_env: Optional[Any] = None) -> Dict[str, float]:
    """Run evaluation on an existing vectorized env and return averaged metrics.

    Accumulates step-wise reward dictionaries from infos into per-episode `paths`:
      path = { 'env_infos': { 'rwd_dict': { key: np.array([...]) } } }

    Returns a dict of averaged metrics across all completed episodes. The provided
    `eval_vec` is reset at the start and reused across calls for performance.
    """
    # Per-env accumulators of rwd_dict over time
    rwd_accum: List[List[Dict[str, Any]]] = [[] for _ in range(eval_vec.num_envs)]
    collected = 0
    all_paths: List[Dict[str, Any]] = []

    obs = eval_vec.reset()
    # Support for Recurrent policies
    states = None
    episode_starts = np.ones((eval_vec.num_envs,), dtype=bool)

    while collected < episodes:
        # Pass states/episode_starts for compatibility with RecurrentPPO.
        # Standard PPO will accept these as kwargs but might ignore them if not used,
        # or we rely on the fact that predict signature is generally (obs, state, ...)
        # We need to capture the returned states for the next step.
        actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
        
        obs, rewards, dones, infos = eval_vec.step(actions)
        episode_starts = dones

        # Accumulate step rwd_dicts
        for i in range(eval_vec.num_envs):
            info_i = infos[i] if isinstance(infos, (list, tuple)) else {}
            rwd_dict = info_i.get('rwd_dict') if isinstance(info_i, dict) else None
            if isinstance(rwd_dict, dict):
                rwd_accum[i].append(rwd_dict)

        # Handle episode terminations
        for i, done in enumerate(dones):
            if done:
                # Build path with stacked rwd_dict arrays
                steps_dicts = rwd_accum[i]
                stacked: Dict[str, np.ndarray] = {}
                if len(steps_dicts) > 0:
                    keys = set().union(*[d.keys() for d in steps_dicts])
                    for k in keys:
                        vals = [d.get(k) for d in steps_dicts]
                        numeric_vals: List[float] = []
                        for v in vals:
                            if isinstance(v, (int, float, np.floating)):
                                numeric_vals.append(float(v))
                            elif isinstance(v, np.ndarray) and v.size == 1:
                                numeric_vals.append(float(v.item()))
                        if len(numeric_vals) > 0:
                            stacked[k] = np.array(numeric_vals, dtype=np.float32)
                path = {'env_infos': {'rwd_dict': stacked}}
                all_paths.append(path)

                # Reset accumulator for this env and increment count
                rwd_accum[i] = []
                collected += 1
                if collected >= episodes:
                    break

    avg_metrics: Dict[str, float] = {}
    
    # Use the provided metrics_env
    try:
        if len(all_paths) > 0 and metrics_env is not None and hasattr(metrics_env, "get_metrics"):
            avg = metrics_env.get_metrics(all_paths)
            if isinstance(avg, dict):
                avg_metrics = {k: float(v) for k, v in avg.items() if isinstance(v, (int, float, np.floating))}
    except Exception:
        pass
        
    return avg_metrics


class PeriodicEvaluator(BaseCallback):
    """Periodically evaluate the current policy and log averaged metrics."""

    def __init__(self, eval_vec: VecMonitor, eval_freq: int = 10000, eval_episodes: int = 10, metrics_env: Optional[Any] = None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_vec = eval_vec
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.metrics_env = metrics_env
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # Align last eval step on resume to prevent immediate long evaluations.
        if self._last_eval_step == 0 and self.num_timesteps > 0:
            self._last_eval_step = self.num_timesteps
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            try:
                t0 = time.time()
                metrics = evaluate_policy(
                    model=self.model,
                    eval_vec=self.eval_vec,
                    episodes=self.eval_episodes,
                    metrics_env=self.metrics_env
                )
                duration_s = time.time() - t0
                if metrics:
                    # Log to W&B with eval/ prefix
                    log_data = {f"eval/{k}": v for k, v in metrics.items()}
                    log_data["eval/duration_s"] = duration_s
                    wandb.log(log_data, step=self.num_timesteps)
                    if self.verbose:
                        print(f"[Eval] step={self.num_timesteps} metrics={metrics} duration_s={duration_s:.2f}")
            except Exception as e:
                if self.verbose:
                    print(f"[Eval] Evaluation failed: {e}")
        return True

    def __del__(self):
        # We don't automatically close eval_vec here as it might be managed externally,
        # but if we owned it (like in the original implementation), we would.
        # Since we accept it in __init__, we assume the caller manages it or we can close it if we want to be safe.
        # Ideally, the creator should close it.
        pass
