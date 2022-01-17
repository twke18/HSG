"""Parse CLI arguments."""

import argparse

from hsg.config.default import config, update_config


def parse_args(description=''):
  """Parse CLI arguments.
  """
  parser = argparse.ArgumentParser(description=description)
  # Misc parameters.
  parser.add_argument('--snapshot_dir', required=True, type=str,
                      help='/path/to/snapshot/dir.')
  parser.add_argument('--save_dir', type=str,
                      help='/path/to/save/dir.')
  parser.add_argument('--cfg_path', required=True, type=str,
                      help='/path/to/specific/config/file.')
  parser.add_argument('--semantic_memory_dir', type=str, default=None,
                      help='/path/to/stored/memory/dir.')
  parser.add_argument('--cam_dir', type=str, default=None,
                      help='/path/to/stored/cam/dir.')
  parser.add_argument('--data_dir', type=str, default=None,
                      help='/root/dir/to/data.')
  parser.add_argument('--data_list', type=str, default=None,
                      help='/path/to/data/list.')
  # Network parameters.
  parser.add_argument('--kmeans_num_clusters', type=str,
                      help='H,W')
  parser.add_argument('--label_divisor', type=int,
                      help=2048)

  args, rest = parser.parse_known_args()

  # Update config with arguments.
  update_config(args.cfg_path)

  args = parser.parse_args()

  return args
