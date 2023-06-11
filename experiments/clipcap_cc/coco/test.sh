#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/CLIP_prefix_caption
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs
IMAGE_DIR=${BASE_DIR}/data/mscoco/images
ANNOS_FN=${BASE_DIR}/data/mscoco/coco_test.json

name=clipcap_cc
task=COCO
ckpt=${CKPT_DIR}/ClipCap/${name}_weights.pt
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

conda activate clip_prefix_caption

cd $CODE_DIR
python Retrieval.py \
    --checkpoint $ckpt \
    --annotations_file ${ANNOS_FN} \
    --images_dir ${IMAGE_DIR} \
    --output_dir $output \
    --output_name ${name} \
    --k_test 256 \
    # --device cpu

conda deactivate
