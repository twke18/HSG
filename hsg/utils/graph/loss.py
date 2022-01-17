"""Utility functions for defining Graph-based Clustering Loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import hsg.utils.graph.common as graph_common
import hsg.utils.general.common as common_utils

EPS = 1e-15


def _rank3_trace(x):
  return torch.einsum('ijj->i', x)


def _rank3_diag(x):
  eye = torch.eye(x.size(1)).type_as(x)
  out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
  return out


def dmon_pool_loss(x, adj, s, mask=None, softmax=False):
  """DMon pooling loss [1].
  
  [1]: Graph Clustering with Graph Neural Networks.

  Args:
    x: A `tensor` of shape `[batch_size, num_nodes, channels]`.
    adj: A `tensor` of shape `[batch_size, num_nodes, num_nodes]`.
    s: A `tensor` of shape `[batch_size, num_nodes, num_clusters]`.
      The softmax does not have to be applied beforehand, since it is
      executed within this method.
    mask: (Optional) A `tensor` of shape `[batch_size, num_nodes]`,
      indicating the valid nodes for each graph.

  Returns:
    dmon_loss: A scalar `Tensor`.
    collapse_loss: A scalar `Tensor`.
  """
  x = x.unsqueeze(0) if x.dim() == 2 else x
  adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
  s = s.unsqueeze(0) if s.dim() == 2 else s

  #(batch_size, num_nodes, _), k = x.size(), s.size(-1)
  batch_size, num_nodes, k = s.shape

  if softmax:
    s = torch.softmax(s, dim=-1)

  if mask is not None:
    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    s = s * mask

  # C^T A C in the paper.
  out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

  # C^T d^T d C in the paper.
  d_flat = torch.einsum('ijk->ij', adj)
  dtd = torch.einsum(
      'bij,bjk->bik', d_flat.unsqueeze(2), d_flat.unsqueeze(1))
  out_deg = torch.matmul(torch.matmul(s.transpose(1, 2), dtd), s)

  # DMon regularization: -1/2m * Tr(C^T A C - 1 / 2m * C^T d^T d C).
  dmon_normalizer = 2 * d_flat.sum(dim=1)
  dmon_numerator = _rank3_trace(out_adj - out_deg / dmon_normalizer.view(-1, 1, 1))
  dmon_loss = 1 - dmon_numerator / dmon_normalizer
  dmon_loss = torch.mean(dmon_loss)

  # Orthogonality regularization.
  ss = torch.matmul(s.transpose(1, 2), s)
  i_s = torch.eye(k).type_as(ss).unsqueeze(0)
  #ortho_loss = torch.norm(
  #    ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
  #    i_s / torch.norm(i_s), dim=(-1, -2))
  #ortho_loss = torch.mean(ortho_loss)

  # Collapse regularization: sqrt(k) / n * |C^T C|_F - 1
  collapse_numerator = torch.norm(s.sum(dim=1), dim=1)
  collapse_denomerator = num_nodes / torch.norm(i_s, dim=(-1, -2))
  collapse_loss = collapse_numerator / collapse_denomerator
  collapse_loss = torch.mean(collapse_loss)

  return dmon_loss, collapse_loss


class DMonLoss(_Loss):
  """DMon clustering loss [1].

  [1]: Graph Clustering with Graph Neural Networks.
  """

  def __init__(self,
               adj_knn=None,
               size_average=None,
               reduce=None,
               reduction='mean'):
    """Initializes DMonLoss class.

    Args:
      knn_graph: A scalar indicates building K-NN affinity matrix.
    """
    super(DMonLoss, self).__init__(size_average, reduce, reduction)
    self._knn = adj_knn

  def __repr__(self):
    return 'DMonLoss(adj_knn={})'.format(self._knn)

  def forward(self, logits, x, x_padding_mask=None, x_segment_labels=None):
    """Compute DMon clustering loss.

    dmon_loss = - 1 / (2*m) * Tr(C^T A C - C^T d^T d C)
    collapse_loss = sqrt(k) / n * \| sum_i C_i^T \|_F

    Args:
      logits: A `tensor` of shape `[batch_size, num_clusters, length]`.
      x: A `tensor` of shape `[batch_size, channels, length]`.
      x_padding_mask: A `tensor` of shape `[batch_size, length]`
        indicates if the corresponding node is padded.
      x_segment_labels: A `tensor` of shape `[batch_size, length]`,
        indicates if the nodes belong to the same segment, could be
        useful for self-supervised learning to restrict K-NN graph
        per image.

    Returns:
      dmon_loss: A scalar `tensor`.
      reg_loss: A scalar `tensor`.
    """
    # Positive affinity for pulling cluster assignment.
    kernel_fn = lambda x: graph_common.exp_inner_product_kernel(x, 5)
    adj = graph_common.affinity_matrix_as_attention(
        x, x_padding_mask, x_segment_labels, self._knn,
        True, True, kernel_fn)
    #adj = (adj + adj.transpose(1, 2)) * 0.5
    dmon_loss, collapse_loss = dmon_pool_loss(
        x.transpose(1, 2),
        adj,
        logits.transpose(1, 2),
        ~x_padding_mask)

    return dmon_loss, collapse_loss


class HierarchicalDMonLoss(_Loss):
  """Hierarchical DMon clustering loss.
  """

  def __init__(self,
               adj_knn=None,
               size_average=None,
               reduce=None,
               reduction='mean'):
    """Initializes DMonLoss class.

    Args:
      knn_graph: A scalar indicates building K-NN affinity matrix.
    """
    super(HierarchicalDMonLoss, self).__init__(size_average, reduce, reduction)
    self._knn = adj_knn

  def __repr__(self):
    return 'DMonLoss(adj_knn={})'.format(self._knn)

  def forward(self, probs, x, padding_masks=None, x_segment_labels=None):
    """Compute DMon clustering loss.

    dmon_loss = - 1 / (2*m) * Tr(C^T A C - C^T d^T d C)
    collapse_loss = sqrt(k) / n * \| sum_i C_i^T \|_F

    Args:
      logits: A list of [`tensor_1`, `tensor_2`, ...], where the
        second (last) dimension indicates the number of clusters
        at current (previous) level. For example, `tensor_1` is
        of shape `[batch_size, num_clusters_1, length]` and 
        `tensor_2` is of shape
        `[batch_size, num_clusters_2, num_clusters_1]`.
      x: A `tensor` of shape `[batch_size, channels, length]`.
      padding_masks: A list of [`tensor_1`, `tensor_2`, ..], where
        second dimension indicates the number of clusters of previous
        level. For example, `tensor_1` if of shape `[batch_size, length]`
        and `tensor_2` is of shape `[batch_size, num_clusters_1]`.
      x_segment_labels: A `tensor` of shape `[batch_size, length]`,
        indicates if the nodes belong to the same segment, could be
        useful for self-supervised learning to restrict K-NN graph
        per image.

    Returns:
      dmon_loss: A scalar `tensor`.
      reg_loss: A scalar `tensor`.
    """
    # Positive affinity for pulling cluster assignment.
    kernel_fn = lambda x: graph_common.exp_inner_product_kernel(x, 5)
    x_padding_mask = None if not padding_masks else padding_masks[0]
    adj = graph_common.affinity_matrix_as_attention(
        x, x_padding_mask, x_segment_labels, self._knn,
        True, True, kernel_fn)

    prev_probs, prev_padding_masks = None, None
    dmon_losses, collapse_losses = [], []
    for cur_probs, cur_padding_masks in zip(probs, padding_masks):
      if prev_probs is not None:
        # Discretize clustering probabilities.
        #prev_labs = torch.argmax(prev_probs, dim=-2)
        #prev_probs_t = common_utils.one_hot(prev_labs,
        #                                    prev_probs.shape[-2])
        prev_probs_t = prev_probs.transpose(1, 2)
        #prev_probs_t = prev_probs_t.float()
        if prev_padding_masks is not None:
          prev_masks = (
            (~prev_padding_masks).unsqueeze(2).to(prev_probs_t.dtype))
          prev_probs_t = prev_probs_t * prev_masks
        #prev_probs = torch.softmax(prev_logits, dim=-2)
        prev_probs = prev_probs_t.transpose(1, 2)
        adj = torch.matmul(torch.matmul(prev_probs, adj), prev_probs_t)
      dmon_loss, collapse_loss = dmon_pool_loss(
          x.transpose(1, 2), # this is dummy.
          adj,
          cur_probs.transpose(1, 2),
          ~cur_padding_masks,
          False)
      dmon_losses.append(dmon_loss)
      collapse_losses.append(collapse_loss)

      prev_probs = cur_probs
      prev_padding_masks = cur_padding_masks

    return dmon_losses, collapse_losses


def ncut_pool_loss(x, adj, s, mask=None):
  """Normalized Cut loss [1].
  
  [1]: Normalized Cut Loss for Weakly-supervised CNN Segmentation.

  ncut_loss: sum_k S_k^T W (1 - S_k) / d^T S_k, where S_k denotes
    the clustering assignment of group k.
  self_loss: - S log S.

  Args:
    x: A `tensor` of shape `[batch_size, num_nodes, channels]`.
    adj: A `tensor` of shape `[batch_size, num_nodes, num_nodes]`.
    s: A `tensor` of shape `[batch_size, num_nodes, num_clusters]`.
      The softmax does not have to be applied beforehand, since it is
      executed within this method.
    mask: (Optional) A `tensor` of shape `[batch_size, num_nodes]`,
      indicating the valid nodes for each graph.

  Returns:
    ncut_loss: A scalar `Tensor`.
    self_loss: A scalar `Tensor`.
  """
  x = x.unsqueeze(0) if x.dim() == 2 else x
  adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
  s = s.unsqueeze(0) if s.dim() == 2 else s

  (batch_size, num_nodes, _), k = x.size(), s.size(-1)

  s = torch.softmax(s, dim=-1)

  if mask is not None:
    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask

  # S^T W (1-S) in the paper.
  out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), 1 - s)

  # d^T S in the paper.
  d_flat = torch.einsum('ijk->ij', adj) # BxN
  d = d_flat.unsqueeze(1) # Bx1xN
  out_deg = torch.matmul(d, s) # Bx1xK

  # NCut regularization: Tr(S^T A (1-S)) / d^T S
  ncut_numerator = torch.diagonal(out_adj, dim1=1, dim2=2) # BxK
  ncut_denomerator = out_deg.squeeze(1) # BxK
  ncut_denomerator += 1e-2 # Avoid numerical instability.
  ncut_loss = torch.sum(ncut_numerator / ncut_denomerator, dim=1)
  ncut_loss = torch.mean(ncut_loss)

  # Self regularization.
  safe_s = torch.clamp(s, min=1e-5, max=1)
  entropy = - safe_s * torch.log(safe_s)
  self_loss = torch.mean(entropy.sum(dim=1))

  return ncut_loss, self_loss


class NCutLoss(_Loss):
  """Normalized Cut loss [1].

  [1]: Normalized Cut Loss for Weakly-supervised CNN Segmentation.
  """

  def __init__(self,
               adj_knn=None,
               size_average=None,
               reduce=None,
               reduction='mean'):
    """Initializes NCut class.

    Args:
      knn_graph: A scalar indicates building K-NN affinity matrix.
    """
    super(NCutLoss, self).__init__(size_average, reduce, reduction)
    self._knn = adj_knn

  def __repr__(self):
    return 'NCutLoss(adj_knn={:.2f})'.format(self._knn)

  def forward(self, logits, x, x_padding_mask=None, x_segment_labels=None):
    """Compute Normalized Cut loss.

    ncut_loss: sum_k S_k^T W (1 - S_k) / d^T S_k, where S_k denotes
      the clustering assignment of group k.
    self_loss: - S log S.

    Args:
      logits: A `tensor` of shape `[batch_size, num_clusters, length]`.
      x: A `tensor` of shape `[batch_size, channels, length]`.
      x_padding_mask: A `tensor` of shape `[batch_size, length]`
        indicates if the corresponding node is padded.
      x_segment_labels: A `tensor` of shape `[batch_size, length]`,
        indicates if the nodes belong to the same segment, could be
        useful for self-supervised learning to restrict K-NN graph
        per image.

    Returns:
      ncut_loss: A scalar `tensor`.
      self_loss: A scalar `tensor`.
    """
    adj = graph_common.affinity_matrix_as_attention(
        x, x_padding_mask, x_segment_labels, self._knn, True, True)
    adj = (adj + adj.transpose(1, 2)) * 0.5
    ncut_loss, self_loss= ncut_pool_loss(
        x.transpose(1, 2),
        adj,
        logits.transpose(1, 2),
        ~x_padding_mask)

    return ncut_loss, self_loss

