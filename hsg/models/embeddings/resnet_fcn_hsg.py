"""Build segmentation model with FCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import hsg.models.utils as model_utils
import hsg.utils.general.common as common_utils
import hsg.utils.segsort.common as segsort_common
from hsg.models.backbones.resnet import ResnetBackbone
from hsg.models.embeddings.base_model import ResnetBase
from hsg.models.embeddings.local_model import LocationColorNetwork
from hsg.models.embeddings.transformer_clusters import TransformerClustering
from hsg.models.heads.position_encoding import PositionEmbeddingLearned
from hsg.models.heads.query_encoding import SimpleQueryEmbedding


class ResnetFcn(ResnetBase):
    
  def __init__(self, backbone_depth, strides, dilations, config):
    """Build FCN using ResNet as backbone network.

    Args:
      backbone_depth: A list of integers indicate the number
        of residual layers in each block.
      strides: A list of intergers indicate the stride.
      dilations: A list of integers indicate the dilations.
      config: An easydict of the network configurations.
    """

    super(ResnetFcn, self).__init__()

    # Build Backbone Network.
    self.resnet_backbone = ResnetBackbone(backbone_depth, strides,
                                          dilations, config)

    # Build Feature Pyramid Network.
    N = len(dilations)
    dim = config.network.embedding_dim
    self.conv_out = nn.Sequential(
        nn.Conv2d(2048, 2048, 1, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
        nn.Conv2d(2048, dim, 1, bias=True))

    # Build Local Feature Network.
    self.lfn = LocationColorNetwork(use_color=False, use_location=True,
                                    norm_color=False, smooth_ksize=None)

    # Build Positional Encoding.
    transformer_hidden_dim = dim
    self.pos_embed = PositionEmbeddingLearned(transformer_hidden_dim // 2)

    # Build Query Encoding.
    self.fine_query_embed = SimpleQueryEmbedding(
        num_queries=config.train.fine_hrchy_clusters,
        hidden_dim=transformer_hidden_dim)
    self.coarse_query_embed = SimpleQueryEmbedding(
        num_queries=config.train.coarse_hrchy_clusters,
        hidden_dim=transformer_hidden_dim)

    # Build Transformer for Clustering.
    self.fine_hrchy_transformer = TransformerClustering(
        num_clusters=config.train.fine_hrchy_clusters,
        d_model=transformer_hidden_dim,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=transformer_hidden_dim*2,
        dropout=0.1)
    self.coarse_hrchy_transformer = TransformerClustering(
        num_clusters=config.train.coarse_hrchy_clusters,
        d_model=transformer_hidden_dim,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=transformer_hidden_dim*2,
        dropout=0.1)

    # Parameters for VMF clustering.
    self.label_divisor = config.network.label_divisor
    self.max_num_clusters = 256
    self.fine_hrchy_clusters = config.train.fine_hrchy_clusters
    self.coarse_hrchy_clusters = config.train.coarse_hrchy_clusters

    self.semantic_ignore_index = config.dataset.semantic_ignore_index

    self.kmeans_num_clusters = config.network.kmeans_num_clusters
    self.kmeans_iterations = config.network.kmeans_iterations

    self.initialize()

  def generate_embeddings(self, datas, targets=None, resize_as_input=False):
    """Feed-forward segmentation model to generate pixel-wise embeddings
    and location & RGB features.

    Args:
      datas: A dict with an entry `image`, which is a 4-D float tensor
        of shape `[batch_size, channels, height, width]`.
      targets: A dict with an entry `semantic_label` and `instance_label`,
        which are 3-D long tensors of shape `[batch_size, height, width]`.
      resize_as_input: enable/disable resize_as_input to upscale the 
        embeddings to the same size as the input image.

    Return:
      A dict with entry `embedding` and `local_feature` of shape
      `[batch_size, channels, height, width]`.
    """

    # Generate embeddings.
    _, _, _, res5 = self.resnet_backbone(datas['image'])

    embeddings = self.conv_out(res5)
    pos_embeddings = self.pos_embed(res5)

    if resize_as_input:
      input_size = datas['image'].shape[-2:]
      embeddings = F.interpolate(
          embeddings, size=input_size, mode='bilinear')
      pos_embeddings = F.interpolate(
          pos_embeddings, size=input_size, mode='bilinear')

    size = embeddings.shape[-2:]
    local_features = self.lfn(datas['image'], size=size)

    return {'embedding': embeddings,
            'local_feature': local_features,
            'position_embedding': pos_embeddings}

  def generate_clusters(self,
                        embeddings,
                        semantic_labels,
                        instance_labels,
                        local_features=None,
                        pos_embeddings=None):
    """Perform clustering within each image. There are several
    steps of clustering:
    1. Spherical K-Means using normalized pixel embeddings.
       The clustering is aligned with instance and semantic
       labels to further partition an image into finer segments.
    2. Using Transformer to group above segments hierarchically.


    Args:
      embeddings: A a 4-D float tensor of shape
        `[batch_size, channels, height, width]`.
      semantic_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      instance_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      local_features: A 4-D float tensor of shape
        `[batch_size, height, width, channels]`.
      pos_embeddings: A 4-D float tensor of shape
        `[batch_size, channels, height, width]`.

    Return:
      A dict with entry:
        - `cluster_embedding` of shape `[total_num_pixels, channels]`.
        - `cluster_embedding_with_loc` of shape `[total_num_pixels, channels]`.
        - `cluster_semantic_label` of shape `[total_num_pixels]`.
        - `cluster_instance_label` of shape `[total_num_pixels]`.
        - `cluster_index` of shape `[total_num_pixels]`.
        - `cluster_batch_index` of shape `[total_num_pixels]`.
        - `nd_prototype` of shape `[batch_size, channels, max_num_clusters]`.
        - `nd_prototype_padding_mask` of shape `[batch_size, max_num_clusters]`.
        - `cluster_index_by_image` of shape `[total_num_pixels]`.
        - `finehrchy_nd_prototype_grouping_label` of shape
            `[batch_size, max_num_clusters]`.
        - `finehrchy_nd_prototype_grouping_centroid` of shape
            `[batch_size, channels, fine_hrchy_clusters]`.
        - `finehrchy_nd_prototype_grouping_logit` of shape
            `[batch_size, max_num_clusters, fine_hrchy_clusters]`.
        - `finehrchy_nd_prototype_encoder_memory` of shape
            `[batch_size, channels, max_num_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_label` of shape
            `[batch_size, fine_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_centroid` of shape
            `[batch_size, channels, coarse_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_logit` of shape
            `[batch_size, fine_hrchy_clusters, coarse_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_encoder_memory` of shape
            `[batch_size, channels, fine_hrchy_clusters]`.
    """
    if semantic_labels is not None and instance_labels is not None:
      labels = semantic_labels * self.label_divisor + instance_labels
      ignore_index = labels.max() + 1
      labels = labels.masked_fill(
          semantic_labels == self.semantic_ignore_index,
          ignore_index)
    else:
      labels = None
      ignore_index = None

    # Step1: Spherical KMeans clustering and align with instance
    # and semantic labels.
    (cluster_embeddings,
     cluster_embeddings_with_loc,
     cluster_labels,
     cluster_indices,
     cluster_batch_indices) = (
       segsort_common.segment_by_kmeans(
           embeddings,
           labels,
           self.kmeans_num_clusters,
           local_features=local_features,
           ignore_index=ignore_index,
           iterations=self.kmeans_iterations))

    cluster_semantic_labels = cluster_labels // self.label_divisor
    cluster_instance_labels = cluster_labels % self.label_divisor

    # Positional embeddings are not normalized, hence the small hack
    # is to select the valid pixels (where semantic_labels != ignore_index).
    if pos_embeddings is not None and labels is not None:
      valid_pixels = (labels != ignore_index).view(-1).nonzero().view(-1)
      flat_pos_embeddings = pos_embeddings.permute(0, 2, 3, 1).contiguous()
      flat_pos_embeddings = flat_pos_embeddings.flatten(0, 2)
      cluster_pos_embeddings = torch.index_select(
          flat_pos_embeddings, 0, valid_pixels)
    else:
      cluster_pos_embeddings = None

    # Step2. Extract KMeans clustering prototypes. Note: the
    # prototype could be lifted to hyper-bolic space from here.
    (prototypes,
     pos_prototypes,
     prototype_padding_masks,
     prototype_labels,
     prototype_batch_indices,
     cluster_indices_by_image) = self._calculate_kmeans_prototypes(
         cluster_embeddings,
         cluster_indices,
         cluster_batch_indices,
         cluster_pos_embeddings,
         cluster_labels)
    prototype_semantic_labels = prototype_labels // self.label_divisor
    prototype_instance_labels = prototype_labels % self.label_divisor

    # Step3: Extract segments and group them hierarchically with
    # Transformer.
    (finehrchy_prototype_grouping_labels,
     finehrchy_prototype_grouping_centroids,
     finehrchy_prototype_grouping_logits,
     finehrchy_prototype_encoder_memory,
     coarsehrchy_prototype_grouping_labels,
     coarsehrchy_prototype_grouping_centroids,
     coarsehrchy_prototype_grouping_logits,
     coarsehrchy_prototype_encoder_memory) = self._hierarchical_grouping(
        prototypes, pos_prototypes, prototype_padding_masks)

    # Collect pixel-wise hierarchical clustering indices.
    finehrchy_cluster_indices = (
      self._collect_pixel_hierarchical_clustering_indices(
          cluster_indices_by_image,
          cluster_batch_indices,
          finehrchy_prototype_grouping_labels))
    coarsehrchy_cluster_indices = (
      self._collect_pixel_hierarchical_clustering_indices(
          cluster_indices_by_image,
          cluster_batch_indices,
          coarsehrchy_prototype_grouping_labels))


    outputs = {
      'cluster_embedding': cluster_embeddings,
      'cluster_embedding_with_loc': cluster_embeddings_with_loc,
      'cluster_semantic_label': cluster_semantic_labels,
      'cluster_instance_label': cluster_instance_labels,
      'cluster_index': cluster_indices,
      'cluster_batch_index': cluster_batch_indices,
      'finehrchy_cluster_index': finehrchy_cluster_indices,
      'coarsehrchy_cluster_index': coarsehrchy_cluster_indices,
      'nd_prototype': prototypes,
      'nd_prototype_padding_mask': prototype_padding_masks,
      'nd_prototype_batch_index': prototype_batch_indices,
      'nd_prototype_semantic_label': prototype_semantic_labels,
      'nd_prototype_instance_label': prototype_instance_labels,
      'cluster_index_by_image': cluster_indices_by_image,
      'finehrchy_nd_prototype_grouping_label': (
        finehrchy_prototype_grouping_labels),
      'finehrchy_nd_prototype_grouping_centroid': (
        finehrchy_prototype_grouping_centroids),
      'finehrchy_nd_prototype_grouping_logit': (
        finehrchy_prototype_grouping_logits),
      'finehrchy_nd_prototype_encoder_memory': (
        finehrchy_prototype_encoder_memory),
      'coarsehrchy_nd_prototype_grouping_label': (
        coarsehrchy_prototype_grouping_labels),
      'coarsehrchy_nd_prototype_grouping_centroid': (
        coarsehrchy_prototype_grouping_centroids),
      'coarsehrchy_nd_prototype_grouping_logit': (
        coarsehrchy_prototype_grouping_logits),
      'coarsehrchy_nd_prototype_encoder_memory': (
        coarsehrchy_prototype_encoder_memory),
    }

    return outputs

  def forward(self, datas, targets=None, resize_as_input=None):
    """Generate pixel-wise embeddings and Spherical Kmeans clustering
    within each image.
    """

    targets = targets if targets is not None else {}

    # Generaet embeddings.
    outputs = self.generate_embeddings(datas, targets, resize_as_input)

    # Resize labels to embedding size.
    semantic_labels = targets.get('semantic_label', None)
    if semantic_labels is not None:
      semantic_labels = common_utils.resize_labels(
          semantic_labels, outputs['embedding'].shape[-2:])

    instance_labels = targets.get('instance_label', None)
    if instance_labels is not None:
      instance_labels = common_utils.resize_labels(
          instance_labels, outputs['embedding'].shape[-2:])

    # Generate clusterings.
    cluster_embeddings = self.generate_clusters(
        outputs['embedding'],
        semantic_labels,
        instance_labels,
        outputs['local_feature'],
        outputs['position_embedding'])

    outputs.update(cluster_embeddings)

    return outputs

  def initialize(self):
    pass

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []
    resnet_params_name = ['resnet_backbone.conv1',
                          'resnet_backbone.res2',
                          'resnet_backbone.res3',
                          'resnet_backbone.res4',
                          'resnet_backbone.res5',]
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          resnet_params_name,
          ['weight'])],
      'lr': 1})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          resnet_params_name,
          ['bias'])],
      'lr': 2,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['conv_out'],
          ['weight'])],
      'lr': 1})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['conv_out'],
          ['bias'])],
      'lr': 2,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['pos_embed'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['pos_embed'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['fine_query_embed'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['fine_query_embed'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['coarse_query_embed'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['coarse_query_embed'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['fine_hrchy_transformer'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['fine_hrchy_transformer'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['coarse_hrchy_transformer'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['coarse_hrchy_transformer'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})

    return ret

  def name_mapping(self, name, resume=False):
    if resume:
      return name if not name.startswith('module.') else name[len('module.'):]

    if name.startswith('conv1') or name.startswith('bn1'):
      return 'resnet_backbone.conv1.' + name

    return name.replace('layer1', 'resnet_backbone.res2')\
               .replace('layer2', 'resnet_backbone.res3')\
               .replace('layer3', 'resnet_backbone.res4')\
               .replace('layer4', 'resnet_backbone.res5')

  def _calculate_kmeans_prototypes(self,
                                   cluster_embeddings,
                                   cluster_indices,
                                   cluster_batch_indices,
                                   cluster_pos_embeddings,
                                   cluster_labels):
    """Extract prototypes from K-Means clustering.

    TODO(twke): Could be used to extract hyperbolic prototypes.

    Args:
      cluster_embeddings: A `tensor` of shape
        `[total_num_pixels, channels]`.
      cluster_pos_embeddings: A `tensor` of shape
        `[total_num_pixels, channels]`.
      cluster_indices: A `tensor` of shape `[total_num_pixels]`,
        where index values are ordered w.r.t whole batch per GPU.
      cluster_batch_indices: A `tensor` of shape `[total_num_pixels]`
      cluster_labels: A `tensor` of shape `[total_num_pixels]`

    Returns:
      prototypes: A `tensor` of shape
        `[batch_size, channels, max_num_clusters]`, where dim-1 is
        padded with 0 values if number of KMeans clusters is less
        than `max_num_clusters`.
      prototype_padding_masks: A `tensor` of shape
        `[batch_size, max_num_clusters]`, where dim-1 is padded with
        `True` value if number of KMeans clusters is less than
        `max_num_clusters`.
      cluster_indices_by_image: A `tensor` of shape
        `[total_num_pixels]`, where index values are ordered w.r.t
        each image.
    """
    # Compute maximum number of clusters per image on each GPU.
    unique_batch_and_cluster_indices = torch.unique(
        cluster_batch_indices * self.label_divisor + cluster_indices)
    max_num_clusters = self.max_num_clusters

    # Extract segment prototypes. Note that `cluster_indices` are
    # ordered w.r.t whole batch on each GPU, and we will reorder
    # them w.r.t each image in the batch. For example,
    # `cluster_indices` was [0, 1, 1, 2, 3, 4, 4, 5, ...], whereas
    # `cluster_batch_indices` was [0, 0, 0, 0, 1, 1, 1, 2, ...].
    # `cluster_indices` is re-ordered [0, 1, 1, 2, 0, 1, 1, 0,...]
    prototypes = []
    pos_prototypes = []
    prototype_labels = []
    prototype_batch_indices = []
    prototype_padding_masks = []
    cluster_indices_by_image = []
    for batch_ind in torch.unique(cluster_batch_indices):
      cur_batch_mask = (cluster_batch_indices == batch_ind).nonzero().view(-1)
      c_inds = torch.gather(cluster_indices, 0, cur_batch_mask)
      c_labs = torch.gather(cluster_labels, 0, cur_batch_mask)

      # hi-jack `c_labs` to carry `cluster_batch_indices`.
      c_batch_inds = torch.gather(cluster_batch_indices, 0, cur_batch_mask)
      c_labs = c_batch_inds * self.label_divisor ** 2 + c_labs

      proto_labs, c_inds = segsort_common.prepare_prototype_labels(
          c_labs, c_inds, c_labs.max() + 1)
      proto_batch_inds = proto_labs // self.label_divisor ** 2
      proto_labs =  proto_labs % self.label_divisor ** 2
      num_protos = proto_labs.shape[0]

      # Calculate feature embedding prototypes.
      embs = torch.index_select(cluster_embeddings, 0, cur_batch_mask)
      protos = segsort_common.calculate_prototypes_from_labels(
          embs, c_inds, max_label=max_num_clusters)
      proto_labs = F.pad(proto_labs,
                         (0, max_num_clusters - num_protos),
                         #mode='constant', value=proto_labs.max() + 1)
                         mode='constant', value=-1)
      proto_batch_inds = F.pad(proto_batch_inds,
                               (0, max_num_clusters - num_protos),
                               mode='constant', value=-1)

      # Generate prototype padding masks.
      proto_padding_mask = torch.zeros(num_protos, dtype=torch.long,
                                       device=proto_labs.device)
      proto_padding_mask = F.pad(proto_padding_mask,
                                 (0, max_num_clusters - num_protos),
                                 mode='constant', value=1)
      proto_padding_mask = proto_padding_mask.bool()

      # Append to list.
      prototypes.append(protos)
      prototype_labels.append(proto_labs)
      prototype_batch_indices.append(proto_batch_inds)
      prototype_padding_masks.append(proto_padding_mask)
      cluster_indices_by_image.append(c_inds)

      # Calculate positional embedding prototypes.
      if cluster_pos_embeddings is not None:
        pos_embs = torch.index_select(cluster_pos_embeddings, 0, cur_batch_mask)
        pos_protos = common_utils.segment_mean(pos_embs, c_inds)
        pos_protos = F.pad(pos_protos,
                           (0, 0, 0, max_num_clusters - num_protos),
                           mode='constant', value=0)
        pos_prototypes.append(pos_protos)

    prototypes = torch.stack(prototypes, 0) # NxMxC
    prototypes = prototypes.permute(0, 2, 1) # NxCxM
    prototype_labels = torch.stack(prototype_labels, 0)
    prototype_batch_indices = torch.stack(prototype_batch_indices, 0)
    if pos_prototypes:
      pos_prototypes = torch.stack(pos_prototypes, 0) # NxMxC
      pos_prototypes = pos_prototypes.permute(0, 2, 1) # NxCxM
    else:
      pos_prototypes = None
    prototype_padding_masks = torch.stack(prototype_padding_masks, 0) # NxM
    cluster_indices_by_image = torch.cat(cluster_indices_by_image, 0)

    return (prototypes, pos_prototypes, prototype_padding_masks,
            prototype_labels, prototype_batch_indices, cluster_indices_by_image)


  def _hierarchical_grouping(self,
                             prototypes,
                             pos_prototypes,
                             prototype_padding_masks):
    """Group segments hierarchically with Transformers.

    Args:
      prototypes: A `tensor` of shape
        `[batch_size, channels, max_num_clusters]`, where dim-1 is
        padded with 0 values if number of KMeans clusters is less
        than `max_num_clusters`.
      pos_prototypes: A `tensor` of shape
        `[batch_size, channels, max_num_clusters]`, where dim-1 is
        padded with 0 values if number of KMeans clusters is less
        than `max_num_clusters`.
      prototype_padding_masks: A `tensor` of shape
        `[batch_size, max_num_clusters]`, where dim-1 is padded with
        `True` value if number of KMeans clusters is less than
        `max_num_clusters`.

    Returns:
      finehrchy_prototype_grouping_labels: A `tensor` of shape
        `[batch_size, max_num_clusters]`, indicates the
        fine-level grouping of K-Means clusters.
      finehrchy_prototype_grouping_centroids: A `tensor` of shape
        `[batch_size, channels, fine_hrchy_clusters]`, indicates
        the centroids of fine-level grouping.
      finehrchy_prototype_grouping_logits: A `tensor` of shape
        `[batch_size, max_num_clusters, fine_hrchy_clusters]`,
        indicates assignment logits of K-Means clusters to
        fine-level grouping centroids.
      finehrchy_prototype_encoder_memory: A `tensor` of shape
        `[batch_size, channels, max_num_clusters]`, indicates
        output from transformer encoder.
      coarsehrchy_prototype_grouping_labels: A `tensor` of shape
        `[batch_size, fine_hrchy_clusters]`, indicates the
        coarse-level grouping of fine-level groupings.
      coarsehrchy_prototype_grouping_centroids: A `tensor` of shape
        `[batch_size, channels, coarse_hrchy_clusters]`, indicates
        the centroids of coarse-level grouping.
      coarsehrchy_prototype_grouping_logits: A `tensor` of shape
        `[batch_size, fine_hrchy_clusters, coarse_hrchy_clusters]`,
        indicates assignment logits of fine-level groupings to
        coarse-level grouping centroids.
      coarsehrchy_prototype_encoder_memory: A `tensor` of shape
        `[batch_size, channels, fine_hrchy_clusters]`, indicates
        output from transformer encoder.
    """
    # Fine-level hierarchical grouping from KMeans clusters.
    fine_query_embed = self.fine_query_embed()
    (finehrchy_prototype_grouping_centroids,
     finehrchy_prototype_grouping_centroid_feats,
     finehrchy_prototype_grouping_logits,
     finehrchy_prototype_encoder_memory) = self.fine_hrchy_transformer(
         src=prototypes, 
         mask=prototype_padding_masks,
         query_embed=fine_query_embed,
         pos_embed=pos_prototypes)
    finehrchy_prototype_grouping_logits = torch.softmax(
        finehrchy_prototype_grouping_logits, dim=1) # BxC1xN
    finehrchy_prototype_grouping_labels = torch.argmax(
        finehrchy_prototype_grouping_logits, dim=1)

    # Fine-level prototypes.
    finehrchy_pos_prototypes = self._collect_nd_coarser_prototype(
        pos_prototypes,
        finehrchy_prototype_grouping_labels,
        prototype_padding_masks,
        num_groups=self.fine_hrchy_clusters,
        normalized=False)

    # Define coarse-level grouping as whole-image representations.
    # Compute coarse-level prototypes as average fine-level prototypes,
    # followed by fully-connected layers.
    coarse_query_embed = self.coarse_query_embed()
    (coarsehrchy_prototype_grouping_centroids,
     coarsehrchy_prototype_grouping_centroid_feats,
     coarsehrchy_prototype_grouping_logits,
     coarsehrchy_prototype_encoder_memory) = self.coarse_hrchy_transformer(
         src=finehrchy_prototype_grouping_centroid_feats,
         mask=None,
         query_embed=coarse_query_embed,
         pos_embed=finehrchy_pos_prototypes)
    coarsehrchy_prototype_grouping_logits = torch.softmax(
        coarsehrchy_prototype_grouping_logits, dim=1) # BxC2xC1

    # Next-level clustering assignments by Bayesian.
    coarsehrchy_prototype_grouping_logits = torch.einsum(
        'bij,bjk->bik',
        coarsehrchy_prototype_grouping_logits,
        finehrchy_prototype_grouping_logits) # BxC2xN
    coarsehrchy_prototype_grouping_labels = torch.argmax(
        coarsehrchy_prototype_grouping_logits, dim=1)

    return (finehrchy_prototype_grouping_labels,
            finehrchy_prototype_grouping_centroids,
            finehrchy_prototype_grouping_logits,
            finehrchy_prototype_encoder_memory,
            coarsehrchy_prototype_grouping_labels,
            coarsehrchy_prototype_grouping_centroids,
            coarsehrchy_prototype_grouping_logits,
            coarsehrchy_prototype_encoder_memory)

  def _collect_nd_coarser_prototype(self,
                                    prototypes,
                                    prototype_grouping_labels,
                                    prototype_padding_masks=None,
                                    num_groups=None,
                                    normalized=True):
    """Collect coraser grouping representations given clusterings.

    Args:
      prototypes: A `tensor` of shape `[batch_size, channels, num_nodes]`.
      prototype_grouping_labels: A `tensor` of shape
        `[batch_size, max_num_clusters]`.
      prototype_padding_masks: A `tensor` of shape
        `[batch_size, num_nodes]`, where `True` value indicates the
        entry is padded.
      num_groups: A `scalar` indicates the number of coarser goupings.
      normalized: Enable/disable `normalized` to conduct l2-norm.

    Returns:
      coarser_prototypes: A `tensor` of shape
        `[batch_size, channels, num_groups]`.
    """
    batch_size, channels, num_nodes = prototypes.shape
    if num_groups is None:
      num_groups = prototype_grouping_labels.max() + 1

    if prototype_padding_masks is None:
      prototype_padding_masks = torch.zeros([batch_size, num_nodes],
                                            dtype=torch.bool,
                                            device=prototypes.device)

    # Add one dummy entry for padded entries.
    padded_num_groups = num_groups + 1
    prototype_grouping_labels = prototype_grouping_labels.masked_fill(
        prototype_padding_masks, padded_num_groups-1)

    # Prepare scatter indices and tensors.
    scatter_indices = prototype_grouping_labels.unsqueeze(2)
    scatter_indices = scatter_indices.expand(batch_size, num_nodes, channels)

    coarser_prototypes = torch.zeros(
        [batch_size, padded_num_groups, channels],
        dtype=prototypes.dtype,
        device=prototypes.device)
    counts = torch.zeros(
        [batch_size, padded_num_groups, channels],
        dtype=prototypes.dtype,
        device=prototypes.device)

    # Scatter.
    prototypes_T = prototypes.permute(0, 2, 1) # BxNxC
    ones = torch.ones_like(prototypes_T)
    coarser_prototypes = (
      coarser_prototypes.scatter_add_(1, scatter_indices, prototypes_T))
    counts = counts.scatter_add_(1, scatter_indices, ones)
    counts = torch.clamp(counts, min=1e-12)
    coarser_prototypes = coarser_prototypes / counts

    coarser_prototypes = coarser_prototypes[:, :-1, :]

    if normalized:
      coarser_prototypes = common_utils.normalize_embedding(coarser_prototypes)

    coarser_prototypes = coarser_prototypes.permute(0, 2, 1) # BxCxM

    return coarser_prototypes


  def _collect_pixel_hierarchical_clustering_indices(
      self,
      cluster_indices_by_batch,
      cluster_batch_indices,
      finehrchy_prototype_grouping_labels):
    """Collect pixel-wise fine- and corase-level clustering indices.

    Args:
      cluster_indices_by_batch: A `tensor` of shape `[total_num_pixels]`.
      cluster_batch_indices: A `tensor` of shape `[total_num_pixels]`.
      finehrchy_nd_prototype_grouping_label: A `tensor` of shape
        `[batch_size, max_num_clusters]`.
      finehrchy_nd_prototype_grouping_label: A `tensor` of shape
        `[batch_size, max_num_clusters]`.

    Returns:
      finehrchy_cluster_indices: A `tensor` of shape `[total_num_pixels]`.
    """
    finehrchy_cluster_indices = []
    for i, batch_ind in enumerate(torch.unique(cluster_batch_indices)):
      cur_batch_mask = (cluster_batch_indices == batch_ind).nonzero().view(-1)
      c_inds = torch.gather(cluster_indices_by_batch, 0, cur_batch_mask)
      fh_inds = torch.gather(
          finehrchy_prototype_grouping_labels[i, :], 0, c_inds)

      finehrchy_cluster_indices.append(fh_inds)

    finehrchy_cluster_indices = torch.cat(finehrchy_cluster_indices, 0)

    return finehrchy_cluster_indices



class MultiviewResnetFcn(ResnetFcn):

  def generate_clusters(self,
                        embeddings,
                        semantic_labels,
                        instance_labels,
                        image_indices,
                        local_features=None,
                        pos_embeddings=None):
    """Perform clustering within each image. There are several
    steps of clustering:
    1. Spherical K-Means using normalized pixel embeddings.
       The clustering is aligned with instance and semantic
       labels to further partition an image into finer segments.
    2. Using Transformer to group above segments hierarchically.


    Args:
      embeddings: A a 4-D float tensor of shape
        `[batch_size, channels, height, width]`.
      semantic_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      instance_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      image_indices: A 1-D long tensor of shape `[batch_size]`.
      local_features: A 4-D float tensor of shape
        `[batch_size, height, width, channels]`.
      pos_embeddings: A 4-D float tensor of shape
        `[batch_size, channels, height, width]`.

    Return:
      A dict with entry:
        - `cluster_embedding` of shape `[total_num_pixels, channels]`.
        - `cluster_embedding_with_loc` of shape `[total_num_pixels, channels]`.
        - `cluster_semantic_label` of shape `[total_num_pixels]`.
        - `cluster_instance_label` of shape `[total_num_pixels]`.
        - `cluster_index` of shape `[total_num_pixels]`.
        - `cluster_batch_index` of shape `[total_num_pixels]`.
        - `nd_prototype` of shape `[batch_size, channels, max_num_clusters]`.
        - `nd_prototype_padding_mask` of shape `[batch_size, max_num_clusters]`.
        - `cluster_index_by_image` of shape `[total_num_pixels]`.
        - `finehrchy_nd_prototype_grouping_label` of shape
            `[batch_size, max_num_clusters]`.
        - `finehrchy_nd_prototype_grouping_centroid` of shape
            `[batch_size, channels, fine_hrchy_clusters]`.
        - `finehrchy_nd_prototype_grouping_logit` of shape
            `[batch_size, max_num_clusters, fine_hrchy_clusters]`.
        - `finehrchy_nd_prototype_encoder_memory` of shape
            `[batch_size, channels, max_num_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_label` of shape
            `[batch_size, fine_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_centroid` of shape
            `[batch_size, channles, coarse_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_grouping_logit` of shape
            `[batch_size, fine_hrchy_clusters, coarse_hrchy_clusters]`.
        - `coarsehrchy_nd_prototype_encoder_memory` of shape
            `[batch_size, channels, fine_hrchy_clusters]`.
    """
    if semantic_labels is not None and instance_labels is not None:
      labels = semantic_labels * self.label_divisor + instance_labels
      ignore_index = labels.max() + 1
      labels = labels.masked_fill(
          semantic_labels == self.semantic_ignore_index,
          ignore_index)
    else:
      labels = None
      ignore_index = None

    # Step1: Spherical KMeans clustering and align with instance
    # and semantic labels.
    (cluster_embeddings,
     cluster_embeddings_with_loc,
     cluster_labels,
     cluster_indices,
     cluster_batch_indices) = (
       segsort_common.segment_by_kmeans(
           embeddings,
           labels,
           self.kmeans_num_clusters,
           local_features=local_features,
           ignore_index=ignore_index,
           iterations=self.kmeans_iterations))

    cluster_semantic_labels = cluster_labels // self.label_divisor
    cluster_instance_labels = cluster_labels % self.label_divisor

    # Positional embeddings are not normalized, hence the small hack
    # is to select the valid pixels (where semantic_labels != ignore_index).
    if pos_embeddings is not None and labels is not None:
      valid_pixels = (labels != ignore_index).view(-1).nonzero().view(-1)
      flat_pos_embeddings = pos_embeddings.permute(0, 2, 3, 1).contiguous()
      flat_pos_embeddings = flat_pos_embeddings.flatten(0, 2)
      cluster_pos_embeddings = torch.index_select(
          flat_pos_embeddings, 0, valid_pixels)
    else:
      cluster_pos_embeddings = None

    # Step2. Extract KMeans clustering prototypes. Note: the
    # prototype could be lifted to hyper-bolic space from here.
    (prototypes,
     pos_prototypes,
     prototype_padding_masks,
     prototype_labels,
     prototype_batch_indices,
     cluster_indices_by_image) = self._calculate_kmeans_prototypes(
         cluster_embeddings,
         cluster_indices,
         cluster_batch_indices,
         cluster_pos_embeddings,
         cluster_labels,
         image_indices)
    prototype_semantic_labels = prototype_labels // self.label_divisor
    prototype_instance_labels = prototype_labels % self.label_divisor

    # Step3: Extract segments and group them hierarchically with
    # Transformer.
    (finehrchy_prototype_grouping_labels,
     finehrchy_prototype_grouping_centroids,
     finehrchy_prototype_grouping_logits,
     finehrchy_prototype_encoder_memory,
     coarsehrchy_prototype_grouping_labels,
     coarsehrchy_prototype_grouping_centroids,
     coarsehrchy_prototype_grouping_logits,
     coarsehrchy_prototype_encoder_memory) = self._hierarchical_grouping(
        prototypes, pos_prototypes, prototype_padding_masks)

    # Collect pixel-wise hierarchical clustering indices.
    if image_indices is not None:
      cluster_image_indices = torch.gather(image_indices, 0, cluster_batch_indices)
    else:
      cluster_image_indices = cluster_batch_indices
    finehrchy_cluster_indices = (
      self._collect_pixel_hierarchical_clustering_indices(
          cluster_indices_by_image,
          cluster_image_indices,
          finehrchy_prototype_grouping_labels))
    coarsehrchy_cluster_indices = (
      self._collect_pixel_hierarchical_clustering_indices(
          cluster_indices_by_image,
          cluster_image_indices,
          coarsehrchy_prototype_grouping_labels))

    outputs = {
      'cluster_embedding': cluster_embeddings,
      'cluster_embedding_with_loc': cluster_embeddings_with_loc,
      'cluster_semantic_label': cluster_semantic_labels,
      'cluster_instance_label': cluster_instance_labels,
      'cluster_index': cluster_indices,
      'cluster_batch_index': cluster_batch_indices,
      'finehrchy_cluster_index': finehrchy_cluster_indices,
      'coarsehrchy_cluster_index': coarsehrchy_cluster_indices,
      'nd_prototype': prototypes,
      'nd_prototype_padding_mask': prototype_padding_masks,
      'nd_prototype_batch_index': prototype_batch_indices,
      'nd_prototype_semantic_label': prototype_semantic_labels,
      'nd_prototype_instance_label': prototype_instance_labels,
      'cluster_index_by_image': cluster_indices_by_image,
      'finehrchy_nd_prototype_grouping_label': (
        finehrchy_prototype_grouping_labels),
      'finehrchy_nd_prototype_grouping_centroid': (
        finehrchy_prototype_grouping_centroids),
      'finehrchy_nd_prototype_grouping_logit': (
        finehrchy_prototype_grouping_logits),
      'finehrchy_nd_prototype_encoder_memory': (
        finehrchy_prototype_encoder_memory),
      'coarsehrchy_nd_prototype_grouping_label': (
        coarsehrchy_prototype_grouping_labels),
      'coarsehrchy_nd_prototype_grouping_centroid': (
        coarsehrchy_prototype_grouping_centroids),
      'coarsehrchy_nd_prototype_grouping_logit': (
        coarsehrchy_prototype_grouping_logits),
      'coarsehrchy_nd_prototype_encoder_memory': (
        coarsehrchy_prototype_encoder_memory),
    }

    return outputs

  def forward(self, datas, targets=None, resize_as_input=None):
    """Generate pixel-wise embeddings and Spherical Kmeans clustering
    within each image.
    """

    targets = targets if targets is not None else {}

    # Generaet embeddings.
    outputs = self.generate_embeddings(datas, targets, resize_as_input)

    # Resize labels to embedding size.
    semantic_labels = targets.get('semantic_label', None)
    if semantic_labels is not None:
      semantic_labels = common_utils.resize_labels(
          semantic_labels, outputs['embedding'].shape[-2:])

    instance_labels = targets.get('instance_label', None)
    if instance_labels is not None:
      instance_labels = common_utils.resize_labels(
          instance_labels, outputs['embedding'].shape[-2:])

    # Generate clusterings.
    cluster_embeddings = self.generate_clusters(
        outputs['embedding'],
        semantic_labels,
        instance_labels,
        targets['image_index'],
        outputs['local_feature'],
        outputs['position_embedding'])

    outputs.update(cluster_embeddings)

    return outputs


  def _calculate_kmeans_prototypes(self,
                                   cluster_embeddings,
                                   cluster_indices,
                                   cluster_batch_indices,
                                   cluster_pos_embeddings,
                                   cluster_labels,
                                   image_indices):
    """Extract prototypes from K-Means clustering. Note that this
    function is different from ResnetFcn._calculate_kmeans_prototypes
    when we consider images and their augmented counterparts. We will
    stack prototypes w.r.t the data with the same image ids.

    TODO(twke): Could be used to extract hyperbolic prototypes.

    Args:
      cluster_embeddings: A `tensor` of shape
        `[total_num_pixels, channels]`.
      cluster_pos_embeddings: A `tensor` of shape
        `[total_num_pixels, channels]`.
      cluster_indices: A `tensor` of shape `[total_num_pixels]`,
        where index values are ordered w.r.t whole batch per GPU.
      cluster_batch_indices: A `tensor` of shape `[total_num_pixels]`
      cluster_labels: A `tensor` of shape `[total_num_pixels]`
      image_indices: A `tensor` of shape `[batch_size]`.

    Returns:
      prototypes: A `tensor` of shape
        `[batch_size // num_aug, channels, max_num_clusters]`,
        where dim-1 is padded with 0 values if number of KMeans clusters
        is less than `max_num_clusters`.
      pos_prototypes: A `tensor` of shape
        `[batch_size // num_aug, channels, max_num_clusters]`,
        where dim-1 is padded with 0 values if number of KMeans clusters
        is less than `max_num_clusters`.
      prototype_padding_masks: A `tensor` of shape
        `[batch_size // num_aug, max_num_clusters]`, where
        dim-1 is padded with `True` value if number of KMeans clusters
        is less than `max_num_clusters`.
      prototype_labels: A `tensor` of shape
        `[batch_size // num_aug, max_num_clusters]`.
      prototype_batch_indices: A `tensor` of shape
        `[batch_size // num_aug, max_num_clusters]`.
      cluster_indices_by_image: A `tensor` of shape
        `[total_num_pixels]`, where index values are ordered w.r.t
        each image.
    """
    # Pair each image with its augmentation.
    if image_indices is not None:
      cluster_image_indices = torch.gather(image_indices, 0, cluster_batch_indices)
    else:
      cluster_image_indices = cluster_batch_indices

    # Compute maximum number of clusters per image on each GPU.
    unique_image_and_cluster_indices = torch.unique(
        cluster_image_indices * self.label_divisor + cluster_indices)
    max_num_clusters = self.max_num_clusters

    # Extract segment prototypes. Note that `cluster_indices` are
    # ordered w.r.t whole batch on each GPU, and we will reorder
    # them w.r.t each image and its augmentation in the batch.
    # For example,
    # `cluster_indices` was [0, 1, 1, 2, 3, 4, 4, 5, ...], whereas
    # `cluster_batch_indices` was [0, 0, 0, 0, 1, 1, 1, 2, ...].
    # corresponding image indices are [0, 0, 0, 0, 0, 0, 0, 1, ...].
    # `cluster_indices` is re-ordered [0, 1, 1, 2, 3, 4, 4, 0,...]
    prototypes = []
    pos_prototypes = []
    prototype_labels = []
    prototype_batch_indices = []
    prototype_padding_masks = []
    cluster_indices_by_image = []
    for img_ind in torch.unique(cluster_image_indices):
      cur_image_mask = (cluster_image_indices == img_ind).nonzero().view(-1)
      c_inds = torch.gather(cluster_indices, 0, cur_image_mask)
      c_labs = torch.gather(cluster_labels, 0, cur_image_mask)
      # hi-jack `c_labs` to carry `cluster_batch_indices`.
      c_batch_inds = torch.gather(cluster_batch_indices, 0, cur_image_mask)
      c_labs = c_batch_inds * self.label_divisor ** 2 + c_labs

      proto_labs, c_inds = segsort_common.prepare_prototype_labels(
          c_labs, c_inds, c_labs.max() + 1)
      proto_batch_inds = proto_labs // self.label_divisor ** 2
      proto_labs =  proto_labs % self.label_divisor ** 2
      num_protos = proto_labs.shape[0]

      # Calculate feature embedding prototypes.
      embs = torch.index_select(cluster_embeddings, 0, cur_image_mask)
      protos = segsort_common.calculate_prototypes_from_labels(
          embs, c_inds, max_label=max_num_clusters)
      proto_labs = F.pad(proto_labs,
                         (0, max_num_clusters - num_protos),
                         mode='constant', value=-1)
      proto_batch_inds = F.pad(proto_batch_inds,
                               (0, max_num_clusters - num_protos),
                               mode='constant', value=-1)

      # Generate prototype padding masks.
      proto_padding_mask = torch.zeros(num_protos, dtype=torch.long,
                                       device=proto_labs.device)
      proto_padding_mask = F.pad(proto_padding_mask,
                                 (0, max_num_clusters - num_protos),
                                 mode='constant', value=1)
      proto_padding_mask = proto_padding_mask.bool()

      # Append to list.
      prototypes.append(protos)
      prototype_labels.append(proto_labs)
      prototype_batch_indices.append(proto_batch_inds)
      prototype_padding_masks.append(proto_padding_mask)
      cluster_indices_by_image.append(c_inds)

      # Calculate positional embedding prototypes.
      if cluster_pos_embeddings is not None:
        pos_embs = torch.index_select(cluster_pos_embeddings, 0, cur_image_mask)
        pos_protos = common_utils.segment_mean(pos_embs, c_inds)
        pos_protos = F.pad(pos_protos,
                           (0, 0, 0, max_num_clusters - num_protos),
                           mode='constant', value=0)
        pos_prototypes.append(pos_protos)

    prototypes = torch.stack(prototypes, 0) # NxMxC
    prototypes = prototypes.permute(0, 2, 1) # NxCxM
    prototype_labels = torch.stack(prototype_labels, 0)
    prototype_batch_indices = torch.stack(prototype_batch_indices, 0)
    if pos_prototypes:
      pos_prototypes = torch.stack(pos_prototypes, 0) # NxMxC
      pos_prototypes = pos_prototypes.permute(0, 2, 1) # NxCxM
    else:
      pos_prototypes = None
    prototype_padding_masks = torch.stack(prototype_padding_masks, 0) # NxM
    cluster_indices_by_image = torch.cat(cluster_indices_by_image, 0)

    return (prototypes, pos_prototypes, prototype_padding_masks,
            prototype_labels, prototype_batch_indices, cluster_indices_by_image)


def resnet_101_fcn(config):
  """FCN with resnet101 backbone.
  """
  return ResnetFcn([3, 4, 23, 3], [1, 2, 2, 1], [1, 1, 1, 2], config)


def resnet_50_fcn(config):
  """FCN with resnet50 backbone.
  """
  return ResnetFcn([3, 4, 6, 3], [1, 2, 2, 1], [1, 1, 1, 2], config)


def resnet_101_fcn_multiview(config):
  """FCN with resnet101 backbone under multi-view settings.
  """
  return MultiviewResnetFcn([3, 4, 23, 3], [1, 2, 2, 1], [1, 1, 1, 2], config)


def resnet_50_fcn_multiview(config):
  """FCN with resnet50 backbone under multi-view settings.
  """
  return MultiviewResnetFcn([3, 4, 6, 3], [1, 2, 2, 1], [1, 1, 1, 2], config)
