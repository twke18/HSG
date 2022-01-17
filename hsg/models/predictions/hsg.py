"""Define HSG for unsupervised hierarchical semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import hsg.models.utils as model_utils
import hsg.utils.general.common as common_utils
import hsg.utils.segsort.loss as segsort_loss
import hsg.utils.graph.loss as graph_loss
import hsg.utils.segsort.eval as segsort_eval
import hsg.utils.segsort.common as segsort_common


class Hsg(nn.Module):

  def __init__(self, config):
    """Initializes Hsg class which defines losses and inference.

    For losses, there are several types of regualarization:
      - pixel-to-segment contrastive loss w.r.t region segmentations.
      - segment-to-segment contrastive loss w.r.t hierarchical groupings.
      - clustering regularizations for transformers.
    """
    super(Hsg, self).__init__()

    # Define regularization by low-level image similarity.
    self.img_sim_loss = self._construct_loss(
        config.train.img_sim_loss_types,
        concentration=config.train.img_sim_concentration)
    self.img_sim_loss_weight = config.train.img_sim_loss_weight

    # Define regularization by fine-level hierarchical grouping.
    self.fine_hrchy_loss = self._construct_loss(
        config.train.fine_hrchy_loss_types,
        concentration=config.train.fine_hrchy_concentration)
    self.fine_hrchy_loss_weight = config.train.fine_hrchy_loss_weight

    # Define regularization by coarse-level hierarchical grouping.
    self.coarse_hrchy_loss = self._construct_loss(
        config.train.coarse_hrchy_loss_types,
        concentration=config.train.coarse_hrchy_concentration)
    self.coarse_hrchy_loss_weight = config.train.coarse_hrchy_loss_weight

    # Define regularization for clustering transformers.
    self.dmon_loss = self._construct_loss(
        config.train.dmon_loss_types,
        adj_knn=config.train.dmon_knn)
    self.dmon_loss_weight = config.train.dmon_loss_weight
    self.centroid_cont_loss = self._construct_loss(
        config.train.centroid_cont_loss_types,
        concentration=config.train.centroid_cont_concentration)
    self.centroid_cont_loss_weight = config.train.centroid_cont_loss_weight

    self.semantic_ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes
    self.label_divisor = config.network.label_divisor

  def _construct_loss(self, loss_types, **kwargs):

    if loss_types == 'segsort':
      return segsort_loss.SegSortLoss(kwargs['concentration'],
                                      group_mode='segsort+',
                                      reduction='mean')
    elif loss_types == 'dmon':
      return graph_loss.DMonLoss(adj_knn=kwargs['adj_knn'])
    elif loss_types == 'none':
      return None
    else:
      raise KeyError('Unsupported loss types: {:s}'.format(loss_types))

  def predictions(self, datas, targets={}):
    """Predict semantic segmentation by Nearest Neighbor Retrievals.
    """
    raise NotImplementedError()

  def losses(self, datas, targets={}):
    """Compute losses.
    """
    img_sim_loss = None
    hrchy_group_loss = None
    clustering_loss = None
    img_sim_acc = None

    # Compute low-level image similarity loss.
    if self.img_sim_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding']
      batch_indices = datas['cluster_batch_index']
      image_indices = torch.gather(
          targets['image_index'], 0, batch_indices)
      instance_labels = datas['cluster_instance_label']
      instance_labels = (
        instance_labels*self.label_divisor + image_indices)

      prototypes = targets['prototype']
      prototype_batch_indices = targets['prototype_batch_index']
      prototype_image_indices = torch.gather(
          targets['image_index'], 0, prototype_batch_indices)
      prototype_instance_labels = targets['prototype_instance_label']
      prototype_instance_labels = (
        prototype_instance_labels*self.label_divisor + prototype_image_indices)

      img_sim_loss = self.img_sim_loss(
          embeddings,
          instance_labels,
          cluster_indices,
          prototypes,
          prototype_instance_labels)
      img_sim_loss *= self.img_sim_loss_weight

      img_sim_acc, _ = segsort_eval.top_k_ranking(
          prototypes,
          prototype_instance_labels,
          prototypes,
          prototype_instance_labels,
          5)

    if self.fine_hrchy_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding']

      prototypes = targets['prototype']

      prototype_finehrchy_labels = targets['finehrchy_mapping_index']
      finehrchy_labels = torch.gather(
          prototype_finehrchy_labels, 0, cluster_indices)

      fine_hrchy_loss = self.fine_hrchy_loss(
          embeddings,
          finehrchy_labels,
          cluster_indices,
          prototypes,
          prototype_finehrchy_labels)
      fine_hrchy_loss *= self.fine_hrchy_loss_weight
      hrchy_group_loss = fine_hrchy_loss

    if self.coarse_hrchy_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding']

      prototypes = targets['prototype']

      prototype_coarsehrchy_labels = targets['coarsehrchy_mapping_index']
      coarsehrchy_labels = torch.gather(
          prototype_coarsehrchy_labels, 0, cluster_indices)

      coarse_hrchy_loss = self.coarse_hrchy_loss(
          embeddings,
          coarsehrchy_labels,
          cluster_indices,
          prototypes,
          prototype_coarsehrchy_labels)
      coarse_hrchy_loss *= self.coarse_hrchy_loss_weight
      if hrchy_group_loss is None:
        hrchy_group_loss = coarse_hrchy_loss
      else:
        hrchy_group_loss += coarse_hrchy_loss

    if self.dmon_loss is not None:
      # Compute affinity w.r.t Transformer Encoder output.
      fine_nd_logits = datas['finehrchy_nd_prototype_grouping_logit']
      coarse_nd_logits = datas['coarsehrchy_nd_prototype_grouping_logit']
      nd_prototypes = datas['nd_prototype']
      nd_prototype_batch_indices = datas['nd_prototype_batch_index']
      nd_prototype_padding_masks = datas['nd_prototype_padding_mask']

      # Compute DMon Loss and Collapse Regularization.
      dmon_losses, collapse_losses = [], []
      for nd_logits in [coarse_nd_logits, fine_nd_logits]:

        dmon_loss, reg_loss = self.dmon_loss(nd_logits,
                                             nd_prototypes,
                                             nd_prototype_padding_masks,
                                             nd_prototype_batch_indices)
        dmon_losses.append(dmon_loss)
        collapse_losses.append(reg_loss)

      clustering_loss = sum(dmon_losses) + sum(collapse_losses)
      clustering_loss *= self.dmon_loss_weight

    # Across-image contrastive loss among centroids.
    if self.centroid_cont_loss is not None:
      centroid_cont_losses = []
      for prefix in ['coarse', 'fine']:
        nd_target_centroids = (
          targets[prefix+'hrchy_nd_prototype_grouping_centroid'])

        target_centroid_shape = nd_target_centroids.shape
        nd_target_centroids = (
          nd_target_centroids.permute(0, 2, 1).contiguous().flatten(0, 1))
        norm_target_centroids = (
          common_utils.normalize_embedding(nd_target_centroids))
        nd_target_centroid_labels = torch.arange(
            nd_target_centroids.shape[0],
            dtype=torch.long,
            device=nd_target_centroids.device)
        nd_target_centroid_labels = nd_target_centroid_labels.view(
            target_centroid_shape[0], target_centroid_shape[2])

        nd_centroids = datas[prefix+'hrchy_nd_prototype_grouping_centroid']
        nd_centroids = nd_centroids.permute(0, 2, 1).contiguous().flatten(0, 1)
        norm_centroids = common_utils.normalize_embedding(nd_centroids)

        image_indices = torch.gather(
            targets['image_index'], 0, datas['cluster_batch_index'])
        img_ind_min = image_indices.min().detach()
        img_ind_max = (image_indices.max() + 1).detach()
        nd_centroid_labels = nd_target_centroid_labels[img_ind_min:img_ind_max]

        cont_loss = self.centroid_cont_loss(norm_centroids,
                                            nd_centroid_labels.view(-1),
                                            nd_centroid_labels.view(-1),
                                            norm_target_centroids,
                                            nd_target_centroid_labels.view(-1))
        centroid_cont_losses.append(cont_loss)

      if clustering_loss is None:
        clustering_loss = (
          sum(centroid_cont_losses) * self.centroid_cont_loss_weight)
      else:
        clustering_loss += (
          sum(centroid_cont_losses) * self.centroid_cont_loss_weight)


    return img_sim_loss, hrchy_group_loss, clustering_loss, img_sim_acc



  def forward(self, datas, targets=None,
              with_loss=True, with_prediction=False):
    """Compute loss and predictions.
    """
    targets = targets if targets is not None else {}
    outputs = {}

    if with_prediction:
      # Predict semantic and instance labels.
      semantic_pred, semantic_score = self.predictions(datas, targets)

      outputs.update({'semantic_prediction': semantic_pred,
                      'semantic_score': semantic_score,})

    if with_loss:
      img_sim_loss, hrchy_group_loss, clustering_loss, img_sim_acc = (
          self.losses(datas, targets))

      accs = [img_sim_acc]
      accs = [_a for _a in accs if _a is not None]
      acc = sum(accs) / (len(accs) + 1e-12)

      outputs.update(
          {'img_sim_loss': img_sim_loss,
           'hrchy_group_loss': hrchy_group_loss,
           'clustering_loss': clustering_loss,
           'accuracy': acc})

    return outputs

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []

    return ret


def hsg(config):
  """Non-paramteric prototype predictor.
  """
  return Hsg(config)
