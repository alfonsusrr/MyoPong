import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm


def compute_SAR(acts, n_syn, save=False):
  """
  Takes activation data and desired n_comp as input and returns/optionally saves the ICA, PCA, and Scaler objects

  acts: np.array; rollout data containing the muscle activations
  n_comp: int; number of synergies to use
  """
  pca = PCA(n_components=n_syn)
  pca_act = pca.fit_transform(acts)

  ica = FastICA(max_iter=2000, tol=1e-2)
  pcaica_act = ica.fit_transform(pca_act)

  normalizer = MinMaxScaler((-1, 1))
  normalizer.fit(pcaica_act)

  if save:
    joblib.dump(ica, 'ica.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(normalizer, 'scaler.pkl')

  return ica, pca, normalizer


# load from activations.npy
activations = numpy.load("activations.npy")

ica, pca, normalizer = compute_SAR(activations, n_syn=120, save=True)
