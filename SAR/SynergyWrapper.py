import numpy as np
from myosuite.utils import gym


class SynNoSynWrapper(gym.ActionWrapper):
  """
  gym.ActionWrapper that reformulates the action space as the combination of a task-general synergy space and a
  task-specific orginal space, and uses this mix to step the environment in the original action space.
  """

  def __init__(self, env, ica, pca, scaler, phi):
    super().__init__(env)
    self.ica = ica
    self.pca = pca
    self.scaler = scaler
    self.weight = phi

    self.syn_act_space = self.pca.components_.shape[0]
    self.no_syn_act_space = env.action_space.shape[0]
    self.full_act_space = self.syn_act_space + self.no_syn_act_space

    self.action_space = gym.spaces.Box(
        low=-1., high=1., shape=(self.full_act_space,), dtype=np.float32)

  def action(self, act):
    syn_action = act[:self.syn_act_space]
    no_syn_action = act[self.syn_act_space:]

    syn_action = self.pca.inverse_transform(
        self.ica.inverse_transform(self.scaler.inverse_transform([syn_action]))
    )[0]

    # In some MyoSuite tasks (e.g. table tennis), the full action vector includes
    # both muscle activations and additional "movement" actuators. The synergy
    # reconstruction typically spans only the muscle dimensions, so its length
    # may be smaller than env.action_space.shape[0].
    #
    # To avoid shape mismatches, we:
    #   - blend synergies and original actions over the overlapping prefix, and
    #   - keep any remaining action entries from the original action unchanged.
    min_len = min(syn_action.shape[0], no_syn_action.shape[0])
    mixed_head = self.weight * syn_action[:min_len] + \
        (1 - self.weight) * no_syn_action[:min_len]

    if no_syn_action.shape[0] > min_len:
      tail = no_syn_action[min_len:]
      final_action = np.concatenate([mixed_head, tail], axis=0)
    else:
      final_action = mixed_head

    return final_action


class SynergyWrapper(gym.ActionWrapper):
  """
  gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
  synergy-exploiting actions back into the original muscle activation space.
  """

  def __init__(self, env, ica, pca, phi):
    super().__init__(env)
    self.ica = ica
    self.pca = pca
    self.scaler = phi

    self.action_space = gym.spaces.Box(
        low=-1., high=1., shape=(self.pca.components_.shape[0],), dtype=np.float32)

  def action(self, act):
    action = self.pca.inverse_transform(
        self.ica.inverse_transform(self.scaler.inverse_transform([act])))
    return action[0]
