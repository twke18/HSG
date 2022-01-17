import torch
import torch.nn as nn
import torch.nn.functional as F

import hsg.models.utils as model_utils
import hsg.utils.segsort.common as segsort_common
import hsg.utils.general.common as common_utils


class SphericalKMeans(nn.Module):
  """Spherical KMeans Clustering.
  """
  def __init__(self,
               K=None,
               sample_factor=None,
               iterations=10,
               concentration=10):
    super(SphericalKMeans, self).__init__()
    assert(K or sample_factor), 'Both K and sample_factor are invalid.'
    self.K = K
    self.sample_factor = sample_factor
    self.iterations = iterations
    self.concentration = concentration

  def __repr__(self):
    return 'SphericalKMeans(K={}, sample_factor={}, concentration={})'.format(
        self.K, self.sample_factor, self.concentration)

  def _kmeans_clustering(self, x, concentration):
    x = x.view(-1, x.shape[-1])
    N = x.shape[0]
    K = self.K if self.K is not None else int(N * self.sample_factor)
    K = max(K, 1)

    init_labels = torch.randint(
        0, K, (N,), dtype=torch.long, device=x.device)
    kmean_labs = (
      segsort_common.kmeans_with_initial_labels(
          x, init_labels, iterations=self.iterations))
    
    return kmean_labs

  def forward(self, embeddings, batch_indices=None):
    if batch_indices is None:
      batch_indices = torch.zeros((embeddings.shape[0],),
                                  dtype=torch.long,
                                  device=embeddings.device)

    kmean_labs = []
    batch_labs = []
    for batch_ind in torch.unique(batch_indices):
      mask = (batch_indices == batch_ind).view(-1).nonzero().view(-1)
      embs = torch.index_select(embeddings, 0, mask)
      labs = self._kmeans_clustering(embs, self.concentration)
      kmean_labs.append(labs)
      batch_labs.append(torch.zeros_like(labs).fill_(batch_ind))
    kmean_labs = torch.cat(kmean_labs, dim=0)
    batch_labs = torch.cat(batch_labs, dim=0)

    return kmean_labs, batch_labs


class SpectralClustering(nn.Module):
  """Spectral Clustering.
  """
  def __init__(self,
               K=None,
               sample_factor=None,
               iterations=10,
               concentration=10):
    super(SpectralClustering, self).__init__()
    assert(K or sample_factor), 'Both K and sample_factor are invalid.'
    self.K = K
    self.sample_factor = sample_factor
    self.iterations = iterations
    self.concentration = concentration

  def __repr__(self):
    return 'SpectralClustering(K={}, sample_factor={}, concentration={})'.format(
        self.K, self.sample_factor, self.concentration)

  def _spectral_clustering(self, x, concentration):
    x = x.view(-1, x.shape[-1])
    N = x.shape[0]
    K = self.K if self.K is not None else int(N * self.sample_factor)
    K = max(K, 1)

    A = torch.mm(x, x.t()).mul_(concentration).exp_()
    # remove self-loop
    not_eye = 1 - torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    A *= not_eye

    D = A.sum(dim=1)
    d = torch.diag(1 / D.sqrt())
    L = torch.diag(D) - A
    norm_L = torch.chain_matmul(d, L, d)
    eig_val, eig_vec = torch.symeig(norm_L, eigenvectors=True)
    eig_vec = common_utils.normalize_embedding(eig_vec[:, :K])

    init_labels = torch.randint(
        0, K, (N,), dtype=torch.long, device=x.device)
    kmean_labs = (
      segsort_common.kmeans_with_initial_labels(
          eig_vec, init_labels, iterations=self.iterations))
    
    return kmean_labs

  def forward(self, embeddings, batch_indices=None):
    if batch_indices is None:
      batch_indices = torch.zeros((embeddings.shape[0],),
                                  dtype=torch.long,
                                  device=embeddings.device)

    spectral_labs = []
    batch_labs = []
    for batch_ind in torch.unique(batch_indices):
      mask = (batch_indices == batch_ind).view(-1).nonzero().view(-1)
      embs = torch.index_select(embeddings, 0, mask)
      labs = self._spectral_clustering(embs, self.concentration)
      spectral_labs.append(labs)
      batch_labs.append(torch.zeros_like(labs).fill_(batch_ind))
    spectral_labs = torch.cat(spectral_labs, dim=0)
    batch_labs = torch.cat(batch_labs, dim=0)

    return spectral_labs, batch_labs
