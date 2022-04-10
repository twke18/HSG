"""
Define Transformer class.

This code is modeified from DETR:
https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class _BatchNorm1d(nn.Module):
  """This class is specific for 3D inputs of shape
  [length, batch_size, channels].
  """
  def __init__(self, num_features, eps=1e-5, momentum=0.1,
               affine=True, track_running_stats=True):
    super(_BatchNorm1d, self).__init__()
    self.norm = nn.BatchNorm1d(num_features=num_features,
                               eps=eps,
                               momentum=momentum,
                               affine=affine,
                               track_running_stats=track_running_stats)

  def forward(self, x):
    x_t = x.transpose(1, 2)
    x_t = self.norm(x_t)
    x_t = x_t.transpose(1, 2)
    return x_t


class Transformer(nn.Module):

  def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
               activation="relu", normalize_before=False,
               return_intermediate_dec=False):
    """Initializes Transformer class.

    Args:
      d_model: A scalar indicates the input channels to Transformer.
      nhead: A scalar indicates the number of heads for Attention.
      num_encoder_layers: A scalar indicates the number of Encoder.
      num_decoder_layers: A scalar indicates the number of Decoder.
      dim_feedforward: A scalar indicates the intermediate channels
        to Transformer.
      dropout: A `float` indicates the dropout rate.
      activation: A string indicates the type of non-linear activation.
      normalize_before: A `bool` indicates if applying normalization first.
      return_intermediate_dec: A `bool` indicates if return intermediate
        results from decoders.
    """
    super().__init__()

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = _BatchNorm1d(d_model) if normalize_before else None
    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    decoder_norm = _BatchNorm1d(d_model)
    self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                      return_intermediate=return_intermediate_dec)

    self.tgt_fc = nn.Sequential(nn.Linear(d_model*2, dim_feedforward, bias=False),
                                nn.BatchNorm1d(dim_feedforward),
                                nn.ReLU(inplace=True),
                                nn.Linear(dim_feedforward, d_model, bias=True))


    self._reset_parameters()

    self.d_model = d_model
    self.nhead = nhead

  def _reset_parameters(self):
    """Initializes model weights.
    """
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, mask, query_embed, pos_embed):
    """Feedforward pass of Transformer.

    Args:
      src: A `tensor` of shape `[batch_size, channels, source_sequence_length]`.
      mask: A bool `tensor` of shape `[batch_size, sequence_length]`.
      query_embed: A `tensor` of shape `[target_sequence_length, channels]`
        or `[batch_size, channels, target_sequence_length]`.
      pos_embed: A `tensor` of shape
        `[batch_size, channels, source_sequence_length]`.

    Returns:
      decoder_output: A `tensor` of shape
        `[batch_size, channels, target_sequence_length]`.
      encoder_memory: A `tensor` of the same shape as `src`.
    """
    # flatten NxCxHxW to HWxNxC
    bs, c, sl = src.shape
    src = src.permute(2, 0, 1)
    if pos_embed is not None:
      pos_embed = pos_embed.permute(2, 0, 1)
    if query_embed.ndim == 2:
      tl = query_embed.shape[0]
      query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
    else:
      _, _, tl = query_embed.shape
      query_embed = query_embed.permute(2, 0, 1)

    #tgt = torch.zeros_like(query_embed)
    encoder_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

    # Compute `tgt` as averaged `encoder_memory`. Take care of `mask`.
    if mask is not None:
      mask_t = (~mask).t().type_as(encoder_memory).unsqueeze(2)
      sum_mask_t = torch.clamp(torch.sum(mask_t, dim=0), min=1)
      masked_encoder_memory = encoder_memory * mask_t
      mean_tgt = torch.sum(masked_encoder_memory, dim=0) / sum_mask_t
      centered_tgt = masked_encoder_memory - mean_tgt.unsqueeze(0)
      var_tgt = torch.sum(torch.pow(centered_tgt, 2), dim=0)
      std_tgt = torch.sqrt(var_tgt / (sum_mask_t + 1))
    else:
      mean_tgt = torch.mean(encoder_memory, dim=0)
      std_tgt = torch.std(encoder_memory, dim=0)

    tgt = self.tgt_fc(torch.cat([mean_tgt, std_tgt], dim=-1))
    tgt = tgt.unsqueeze(0).repeat(tl, 1, 1)
    decoder_output = self.decoder(tgt, encoder_memory,
                                  memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=query_embed)

    encoder_memory = encoder_memory.permute(1, 2, 0).view(bs, c, sl)
    decoder_output = decoder_output.permute(1, 2, 0).view(bs, c, tl)
    return decoder_output, encoder_memory


class TransformerEncoder(nn.Module):

  def __init__(self, encoder_layer, num_layers, norm=None):
    super().__init__()
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm

  def forward(self, src,
              mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None):
    output = src

    for layer in self.layers:
      output = layer(output, src_mask=mask,
                     src_key_padding_mask=src_key_padding_mask, pos=pos)

    if self.norm is not None:
      output = self.norm(output)

    return output


class TransformerDecoder(nn.Module):

  def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm
    self.return_intermediate = return_intermediate

  def forward(self, tgt, memory,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None,
              query_pos: Optional[Tensor] = None):
    output = tgt

    intermediate = []

    for layer in self.layers:
      output = layer(output, memory, tgt_mask=tgt_mask,
                     memory_mask=memory_mask,
                     tgt_key_padding_mask=tgt_key_padding_mask,
                     memory_key_padding_mask=memory_key_padding_mask,
                     pos=pos, query_pos=query_pos)
      if self.return_intermediate:
        intermediate.append(self.norm(output))

    if self.norm is not None:
      output = self.norm(output)
      if self.return_intermediate:
        intermediate.pop()
        intermediate.append(output)

    if self.return_intermediate:
      return torch.stack(intermediate)

    return output


class TransformerEncoderLayer(nn.Module):

  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
               activation="relu", normalize_before=False):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = _BatchNorm1d(d_model)
    self.norm2 = _BatchNorm1d(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.activation = _get_activation_fn(activation)
    self.normalize_before = normalize_before

  def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

  def forward_post(self,
                   src,
                   src_mask: Optional[Tensor] = None,
                   src_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None):
    q = k = self.with_pos_embed(src, pos)
    src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src

  def forward_pre(self, src,
                  src_mask: Optional[Tensor] = None,
                  src_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None):
    src2 = self.norm1(src)
    q = k = self.with_pos_embed(src2, pos)
    src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src2 = self.norm2(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
    src = src + self.dropout2(src2)
    return src

  def forward(self, src,
              src_mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None):
    if self.normalize_before:
      return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
    return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
               activation="relu", normalize_before=False):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = _BatchNorm1d(d_model)
    self.norm2 = _BatchNorm1d(d_model)
    self.norm3 = _BatchNorm1d(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = _get_activation_fn(activation)
    self.normalize_before = normalize_before

  def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

  def forward_post(self, tgt, memory,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None,
                   query_pos: Optional[Tensor] = None):
    q = k = self.with_pos_embed(tgt, query_pos)
    tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt

  def forward_pre(self, tgt, memory,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None,
                  query_pos: Optional[Tensor] = None):
    tgt2 = self.norm1(tgt)
    q = k = self.with_pos_embed(tgt2, query_pos)
    tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt2 = self.norm2(tgt)
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt2 = self.norm3(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    tgt = tgt + self.dropout3(tgt2)
    return tgt

  def forward(self, tgt, memory,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None,
              query_pos: Optional[Tensor] = None):
    if self.normalize_before:
      return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                              tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return F.relu
  if activation == "gelu":
    return F.gelu
  if activation == "glu":
    return F.glu
  raise RuntimeError("activation should be relu/gelu, not {activation}.")

