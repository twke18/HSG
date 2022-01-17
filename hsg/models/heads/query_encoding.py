"""Define Query embedding for Transformer."""

import torch
import torch.nn as nn


class SimpleQueryEmbedding(nn.Module):

  def __init__(self, num_queries, hidden_dim):
    """Initializes SimpleQueryEmbedding class.

    Args:
      num_queries: A scalar indicates the number of queries.
      hidden_dim: A scalar indicates the hidden channels.
    """
    super().__init__()
    self._query_embedding = nn.Embedding(num_queries, hidden_dim)

  def forward(self):
    return self._query_embedding.weight
