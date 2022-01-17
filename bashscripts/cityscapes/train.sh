#!/bin/bash
# This script is used for training, inference and benchmarking
# the baseline method with HSG on Cityscapes without
# ground-truth annotations. Users could also modify from this
# script for their use case.
#
# Usage:
#   # From HSG/ directory.
#   source bashscripts/cityscapes/train.sh
#
#

# Set up parameters for network.
BACKBONE_TYPES=fcn_50_hsg_cs
EMBEDDING_DIM=128

# Set up parameters for training.
PREDICTION_TYPES=hsg_cs
TRAIN_SPLIT=train
GPUS=0,1,2,3,4,5,6,7
LR_POLICY=step
USE_SYNCBN=true
SNAPSHOT_STEP=38000
MAX_ITERATION=38000
WARMUP_ITERATION=100
LR=1e-1
WD=1e-4
BATCH_SIZE=4
CROP_SIZE=448
MEMORY_BANK_SIZE=0
KMEANS_ITERATIONS=15
KMEANS_NUM_CLUSTERS=4
IMG_SIM_LOSS_TYPES=segsort # segsort / none
FINE_HRCHY_LOSS_TYPES=segsort # segsort / none
COARSE_HRCHY_LOSS_TYPES=segsort # segsort / none
DMON_LOSS_TYPES=dmon # dmon / none
CENTROID_CONT_LOSS_TYPES=segsort # segsort / none
CONCENTRATION=16
IMG_SIM_LOSS_WEIGHT=1.0
FINE_HRCHY_LOSS_WEIGHT=0.1
COARSE_HRCHY_LOSS_WEIGHT=0.1
DMON_LOSS_WEIGHT=0.2
CENTROID_CONT_LOSS_WEIGHT=0.2
FINE_HRCHY_CLUSTERS=8
COARSE_HRCHY_CLUSTERS=4
DMON_KNN=4

# Set up parameters for inference.
INFERENCE_SPLIT=val
INFERENCE_IMAGE_SIZE=2048
INFERENCE_CROP_SIZE_H=1024
INFERENCE_CROP_SIZE_W=2048
INFERENCE_STRIDE=2048

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/cityscapes/fcn_res50_hsg
echo ${SNAPSHOT_DIR}

# Set up the procedure pipeline.
IS_CONFIG_1=1
IS_TRAIN_1=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory and file list.
DATAROOT=/home/twke/data/Cityscapes
TRAIN_DATA_LIST=datasets/cityscapes/${TRAIN_SPLIT}_pmi.txt
TEST_DATA_LIST=datasets/cityscapes/${INFERENCE_SPLIT}.txt
MEMORY_DATA_LIST=datasets/cityscapes/${TRAIN_SPLIT}_pmi.txt


# Build configuration file for training embedding network.
if [ ${IS_CONFIG_1} -eq 1 ]; then
  if [ ! -d ${SNAPSHOT_DIR}/stage1 ]; then
    mkdir -p ${SNAPSHOT_DIR}/stage1
  fi

  sed -e "s/TRAIN_SPLIT/${TRAIN_SPLIT}/g"\
    -e "s/BACKBONE_TYPES/${BACKBONE_TYPES}/g"\
    -e "s/PREDICTION_TYPES/${PREDICTION_TYPES}/g"\
    -e "s/EMBEDDING_DIM/${EMBEDDING_DIM}/g"\
    -e "s/GPUS/${GPUS}/g"\
    -e "s/BATCH_SIZE/${BATCH_SIZE}/g"\
    -e "s/LABEL_DIVISOR/2048/g"\
    -e "s/USE_SYNCBN/${USE_SYNCBN}/g"\
    -e "s/LR_POLICY/${LR_POLICY}/g"\
    -e "s/SNAPSHOT_STEP/${SNAPSHOT_STEP}/g"\
    -e "s/MAX_ITERATION/${MAX_ITERATION}/g"\
    -e "s/WARMUP_ITERATION/${WARMUP_ITERATION}/g"\
    -e "s/LR/${LR}/g"\
    -e "s/WD/${WD}/g"\
    -e "s/MEMORY_BANK_SIZE/${MEMORY_BANK_SIZE}/g"\
    -e "s/KMEANS_ITERATIONS/${KMEANS_ITERATIONS}/g"\
    -e "s/KMEANS_NUM_CLUSTERS/${KMEANS_NUM_CLUSTERS}/g"\
    -e "s/TRAIN_CROP_SIZE/${CROP_SIZE}/g"\
    -e "s/TEST_SPLIT/${INFERENCE_SPLIT}/g"\
    -e "s/TEST_IMAGE_SIZE/${INFERENCE_IMAGE_SIZE}/g"\
    -e "s/TEST_CROP_SIZE_H/${INFERENCE_CROP_SIZE_H}/g"\
    -e "s/TEST_CROP_SIZE_W/${INFERENCE_CROP_SIZE_W}/g"\
    -e "s/TEST_STRIDE/${INFERENCE_STRIDE}/g"\
    -e "s#PRETRAINED#${PRETRAINED}#g"\
    -e "s/IMG_SIM_LOSS_TYPES/${IMG_SIM_LOSS_TYPES}/g"\
    -e "s/FINE_HRCHY_LOSS_TYPES/${FINE_HRCHY_LOSS_TYPES}/g"\
    -e "s/COARSE_HRCHY_LOSS_TYPES/${COARSE_HRCHY_LOSS_TYPES}/g"\
    -e "s/DMON_LOSS_TYPES/${DMON_LOSS_TYPES}/g"\
    -e "s/CENTROID_CONT_LOSS_TYPES/${CENTROID_CONT_LOSS_TYPES}/g"\
    -e "s/IMG_SIM_CONCENTRATION/${CONCENTRATION}/g"\
    -e "s/FINE_HRCHY_CONCENTRATION/${CONCENTRATION}/g"\
    -e "s/COARSE_HRCHY_CONCENTRATION/${CONCENTRATION}/g"\
    -e "s/CENTROID_CONT_CONCENTRATION/${CONCENTRATION}/g"\
    -e "s/IMG_SIM_LOSS_WEIGHT/${IMG_SIM_LOSS_WEIGHT}/g"\
    -e "s/FINE_HRCHY_LOSS_WEIGHT/${FINE_HRCHY_LOSS_WEIGHT}/g"\
    -e "s/COARSE_HRCHY_LOSS_WEIGHT/${COARSE_HRCHY_LOSS_WEIGHT}/g"\
    -e "s/DMON_LOSS_WEIGHT/${DMON_LOSS_WEIGHT}/g"\
    -e "s/CENTROID_CONT_LOSS_WEIGHT/${CENTROID_CONT_LOSS_WEIGHT}/g"\
    -e "s/FINE_HRCHY_CLUSTERS/${FINE_HRCHY_CLUSTERS}/g"\
    -e "s/COARSE_HRCHY_CLUSTERS/${COARSE_HRCHY_CLUSTERS}/g"\
    -e "s/DMON_KNN/${DMON_KNN}/g"\
    configs/cityscapes_template.yaml > ${SNAPSHOT_DIR}/stage1/config.yaml

  cat ${SNAPSHOT_DIR}/stage1/config.yaml
fi


# Train for the embedding.
if [ ${IS_TRAIN_1} -eq 1 ]; then
  python3 pyscripts/train/train.py\
    --data_dir ${DATAROOT}\
    --data_list ${TRAIN_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --cfg_path ${SNAPSHOT_DIR}/stage1/config.yaml

  python3 pyscripts/inference/prototype.py\
    --data_dir ${DATAROOT}\
    --data_list ${MEMORY_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${TRAIN_SPLIT}\
    --kmeans_num_clusters 12,24\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/stage1/config.yaml

  python3 pyscripts/inference/inference.py\
    --data_dir ${DATAROOT}\
    --data_list ${TEST_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}\
    --semantic_memory_dir ${SNAPSHOT_DIR}/stage1/results/${TRAIN_SPLIT}/semantic_prototype\
    --kmeans_num_clusters 12,24\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/stage1/config.yaml

  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage1/results/val/semantic_gray\
    --gt_dir ${DATAROOT}/gtFineId/val/all\
    --string_replace leftImg8bit,gtFineId_labelIds\
    --num_classes 19

fi
