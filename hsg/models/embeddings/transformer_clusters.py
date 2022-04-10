"""Define Transformer for Clustering."""
import math

import torch
import torch.nn as nn

import hsg.models.utils as model_utils
from hsg.models.heads.transformer import Transformer


class TransformerClustering(nn.Module):

  def __init__(self,
               num_clusters=4,
               d_model=512,
               nhead=8,
               num_encoder_layers=6,
               num_decoder_layers=6,
               dim_feedforward=2048,
               dropout=0.1,
               activation="relu",
               normalize_before=False,
               return_intermediate_dec=False):
    """Initializes a Transformer for Clustering.

    Args:
      num_clusters: A scalar indicates the number of centroids.
      d_model: A scalar indicates the input channels to Transformer.
      nhead: A scalar indicates the number of heads for Attention.
      num_encoder_layers: A scalar indicates the number of Encoder.
      num_decoder_layers: A scalar indicates the number of Decoder.
      d_model: A scalar indicates the intermediate channels to Transformer.
      dropout: A `float` indicates the dropout rate.
      activation: A string indicates the type of non-linear activation.
      normalize_before: A `bool` indicates if applying normalization first.
      return_intermediate_dec: A `bool` indicates if return intermediate
        results from decoders.
    """
    super().__init__()
    self._transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        normalize_before=normalize_before,
        return_intermediate_dec=return_intermediate_dec)
    self.centroid_fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(d_model, d_model, bias=False),
        nn.BatchNorm1d(d_model))
    self.centroid_feat_fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(d_model, d_model, bias=False),
        nn.BatchNorm1d(d_model))
    self._num_clusters = num_clusters

  def forward(self, src, mask, query_embed, pos_embed):
    """Feedforward for clustering with Transformer.

    Args:
      src: A `tensor` of shape `[batch_size, channels, source_sequence_length]`.
      mask: A bool `tensor` of shape `[batch_size, sequence_length]`.
      query_embed: A `tensor` of shape `[target_sequence_length, channels]`
        or `[batch_size, channels, target_sequence_length]`.
      pos_embed: A `tensor` of shape
        `[batch_size, channels, source_sequence_length]`.

    Returns:
      centroids: A `tensor` of shape `[batch_size, channels, num_clusters]`.
      logits: A `tensor` of shape
        `[batch_size, num_clusters, source_sequence_length]`.
      node_features: A `tensor` of shape
        `[batch_size, channels, source_sequence_length]`.
    """
    bs, cs, sl = src.shape
    # `centroids` is of shape `[batch_size, channels, target_sequence_length]`.
    # `node_features` is of shape `[batch_size, channels, source_sequence_length]`.
    centroids, node_features = self._transformer(src,
                                                 mask,
                                                 query_embed,
                                                 pos_embed)
    tl = centroids.shape[-1]
    flat_centroids = centroids.transpose(1, 2).flatten(0, 1)
    centroids = self.centroid_fc(flat_centroids).view(bs, tl, cs).transpose(1, 2)
    centroid_feats = self.centroid_feat_fc(flat_centroids).view(bs, tl, cs).transpose(1, 2)

    # Pick centroids with maximum activations w.r.t `node_features`, where
    # `logits` is of shape
    # `[batch_size, target_sequence_length, source_sequence_length].
    logits = torch.einsum(
        'bij,bjk->bik', centroids.transpose(1, 2), node_features)
    logits = logits / math.sqrt(cs)
    max_logits = torch.max(logits, dim=-1)[0]
    _, topk_inds = torch.topk(max_logits, self._num_clusters, dim=-1)

    centroids = torch.gather(centroids.transpose(1, 2),
                             1,
                             topk_inds.unsqueeze(2).repeat(1, 1, cs))
    centroids = centroids.permute(0, 2, 1)
    centroid_feats = torch.gather(centroid_feats.transpose(1, 2),
                                  1,
                                  topk_inds.unsqueeze(2).repeat(1, 1, cs))
    centroid_feats = centroid_feats.permute(0, 2, 1)



    logits = torch.gather(logits,
                          1,
                          topk_inds.unsqueeze(2).repeat(1, 1, sl))

    return centroids, centroid_feats, logits, node_features
