"""Classes for Unsupervised Segmentation Dataset.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as torch_transforms
import PIL.Image as Image

from hsg.data.datasets.base_dataset import ListDataset
import hsg.data.transforms as transforms


class UnsupListDataset(ListDataset):

  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               training=False):

    super(UnsupListDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)

  def _get_datas_by_index(self, idx):
    """Return image_path, semantic_label_path, instance_label_path
    by the given index.
    """
    image_path = self.image_paths[idx]
    image = self._read_image(image_path)

    h, w = image.shape[:2]
    # Do not use ground-truth semantic annotations.
    semantic_label = np.zeros((h, w), dtype=np.int32)

    if len(self.instance_label_paths) > 0:
      instance_label_path = self.instance_label_paths[idx]
      instance_label = self._read_label(instance_label_path)
    else:
      instance_label = None

    return image, semantic_label, instance_label


class SelfsupListDataset(ListDataset):

  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               random_grayscale=True,
               random_blur=False,
               training=False):

    super(SelfsupListDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)
    self.random_grayscale = random_grayscale
    self.random_blur = random_blur

  def _get_datas_by_index(self, idx):
    """Return image_path, semantic_label_path, instance_label_path
    by the given index.
    """
    image_path = self.image_paths[idx]
    image = self._read_image(image_path)

    h, w = image.shape[:2]
    # Do not use ground-truth semantic annotations.
    semantic_label = np.zeros((h, w), dtype=np.int32)

    if len(self.instance_label_paths) > 0:
      instance_label_path = self.instance_label_paths[idx]
      instance_label = self._read_label(instance_label_path)
    else:
      instance_label = None

    return image, semantic_label, instance_label

  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    image, semantic_label, instance_label = self._get_datas_by_index(idx)


    def _data_aug(image, semantic_label, instance_label):
      # Helper function for data processing.
      label = np.stack([semantic_label, instance_label], axis=2)

      if self.random_mirror:
        image, label = transforms.random_mirror(image, label)

      if self.random_scale:
        image, label = transforms.random_resize(image, label, 0.5, 1.5)

      if self.random_crop:
        image, label = transforms.random_crop_with_pad(
            image, label, self.size, self.img_mean, 255)

      # Randomly convert RGB to grayscale.
      if self.random_grayscale:
        image = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
        image = torch_transforms.Compose([
          torch_transforms.RandomApply([
              torch_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
          torch_transforms.RandomGrayscale(p=0.2),
        ])(image)
        image = np.array(image).astype(np.float32) / 255.

      # Randomly apply Gaussian blur.
      if self.random_blur and np.random.uniform(0, 1.0) < 0.5:
        sigma = np.random.uniform(0.1, 5)
        w_x, w_y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        weight = np.exp(- (w_x ** 2 + w_y ** 2) / sigma**2)
        weight = weight / weight.sum()
        image = cv2.filter2D(image, -1, weight)

      semantic_label, instance_label = label[..., 0], label[..., 1]

      return image, semantic_label, instance_label

    img, sem_lab, inst_lab = _data_aug(image, semantic_label, instance_label)
    img_t, sem_lab_t, inst_lab_t = _data_aug(image, semantic_label, instance_label)

    return img, sem_lab, inst_lab, img_t, sem_lab_t, inst_lab_t

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    if self.training:
      (image, semantic_label, instance_label,
       image_t, semantic_label_t, instance_label_t) = self._training_preprocess(idx)
    else:
      raise NotImplementedError()

    image = image - np.array(self.img_mean, dtype=image.dtype)
    image = image / np.array(self.img_std, dtype=image.dtype)
    image_t = image_t - np.array(self.img_mean, dtype=image.dtype)
    image_t = image_t / np.array(self.img_std, dtype=image.dtype)

    image = np.stack([image.transpose(2, 0, 1),
                      image_t.transpose(2, 0, 1)], axis=0)
    semantic_label = np.stack([semantic_label, semantic_label_t], axis=0)
    instance_label = np.stack([instance_label, instance_label_t], axis=0)

    inputs = {'image': image}
    labels = {'semantic_label': semantic_label,
              'instance_label': instance_label,
              'image_id': np.array([idx, idx], dtype=np.int32)}

    return inputs, labels, idx

  def collate_fn(self, batch):
    """Customized collate function to group datas into batch.
    """
    images, labels, indices = zip(*batch)

    images = self._collate_fn_dict_list(images)
    labels = self._collate_fn_dict_list(labels)
    indices = torch.LongTensor(indices)

    shape = images['image'].shape
    images['image'] = (
      images['image'].view(-1, shape[-3], shape[-2], shape[-1]))
    shape = labels['semantic_label'].shape
    labels['semantic_label'] = (
      labels['semantic_label'].view(-1, shape[-2], shape[-1]))
    shape = labels['instance_label'].shape
    labels['instance_label'] = (
      labels['instance_label'].view(-1, shape[-2], shape[-1]))
    labels['image_id'] = labels['image_id'].view(-1)

    return images, labels, indices
