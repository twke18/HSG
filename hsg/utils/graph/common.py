"""Utility functions for Graph-based operations.
"""
import math

import torch


def inner_product_kernel(x):
  """Kernel function for computing inner product. Consider the
  last 2 dimensions for inner production.

  sim(i, j) = x_i^T x_j

  Args:
    x: A `tensor` of shape `[D(0), ..., D(N-2), D(N-1), D(N)]`.

  Returns:
    sim: A `tensor` of shape `[D(0), ..., D(N-2), D(N), D(N)]`.
  """
  return torch.einsum('...ij,...jk->...ik', x.transpose(-2, -1), x)


def exp_inner_product_kernel(x, concentration=5):
  """Kernel function for computing inner product. Consider the
  last 2 dimensions for inner production.

  sim(i, j) = exp(concentration * x_i^T x_j)

  Args:
    x: A `tensor` of shape `[D(0), ..., D(N-2), D(N-1), D(N)]`.

  Returns:
    sim: A `tensor` of shape `[D(0), ..., D(N-2), D(N), D(N)]`.
  """
  sim = inner_product_kernel(x)
  return sim.mul(concentration).exp()


def affinity_matrix_as_attention(x,
                                 x_padding_mask=None,
                                 x_segment_labels=None,
                                 knn=None,
                                 remove_self_loop=True,
                                 binarize=True,
                                 kernel_fn=exp_inner_product_kernel):
  """Compute affinity matrix as attention[1].

  A = exp(X X^T / sqrt(d)).

  [1]: Attention is All You Need. NeuRIPS 2017.

  Args:
    x: A `tensor` of shape `[batch_size, channels, num_nodes]`.
    x_padding_mask: A `tensor` of shape `[batch_size, length]`
      indicates if the corresponding node is padded.
    x_segment_labels: A `tensor` of shape `[batch_size, length]`,
      indicates if the nodes belong to the same segment, could be
      useful for self-supervised learning to restrict K-NN graph
      per image.
    knn: A scalar for building K-NN graph.
    remove_self_loop: Enable / disable `remove_self_loop` to enforce
      0 affinity value of node itself.
    binarize: Enable / disable `binarize` to enforce binary affinity
      value.
  """
  batch_size, channels, num_nodes = x.shape
  #A = torch.einsum('bij,bjk->bik', x.transpose(1, 2), x)
  #A = A / math.sqrt(channels)
  #A = A * 5
  #A = torch.exp(A)
  A = kernel_fn(x)

  if x_padding_mask is None:
    x_padding_mask = torch.zeros([batch_size, num_nodes],
                                 dtype=torch.bool, device=A.device)
  if x_segment_labels is None:
    x_segment_labels = torch.zeros([batch_size, num_nodes],
                                   dtype=torch.long, device=A.device)

  # Mask out when one node is invalid.
  xx_padding_mask = x_padding_mask.unsqueeze(-1) | x_padding_mask.unsqueeze(-2)
  A = A.masked_fill(xx_padding_mask, 0)

  # Remove self loop if there are more than one node in the graph.
  if remove_self_loop:
    is_self = torch.eye(num_nodes, dtype=torch.bool, device=A.device)
    is_self = is_self.unsqueeze(0)

    num_valid_nodes = torch.sum((~x_padding_mask).float(), dim=-1)
    is_remove = num_valid_nodes > 1
    is_remove = is_remove.view(-1, 1, 1)

    A = A.masked_fill(is_self & is_remove, 0)

  # Consider K-NN affinity within each segment.
  if knn is not None:
    new_A = []
    x_not_padding_mask = ~x_padding_mask
    # Iterate through each batch. TODO(twke): ugly, should re-factor.
    for batch_ind in range(A.shape[0]):
      cur_segment_labels = x_segment_labels[batch_ind]
      cur_not_padding_mask = x_not_padding_mask[batch_ind]
      cur_A = A[batch_ind]
      # Iterate through each segment.
      for seg_lab in torch.unique(cur_segment_labels):
        mask = cur_not_padding_mask & (cur_segment_labels == seg_lab)
        if not mask.any():
            continue
        k = min(mask.nonzero().view(-1).shape[0], knn)
        mask = mask.view(1, -1).expand(num_nodes, -1)

        # Select top-k values per segment.
        adj = cur_A.masked_select(mask).view(num_nodes, -1)
        topk_val, _ = torch.topk(adj, k, dim=1)
        kth_val = topk_val[:, -1]
        mask = mask & (cur_A < kth_val.view(-1, 1))
        cur_A = cur_A.masked_fill(mask, 0)
      new_A.append(cur_A)
    A = torch.stack(new_A, 0)

  # Binarize affinity value.
  if binarize:
    A = torch.where(A > 0, torch.ones_like(A), torch.zeros_like(A))

  return A
