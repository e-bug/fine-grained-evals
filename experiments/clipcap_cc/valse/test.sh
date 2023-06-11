#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/CLIP_prefix_caption
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs
DATA_DIR=${BASE_DIR}/data/VALSE/data

name=clipcap_cc
task=SVO
ckpt=${CKPT_DIR}/ClipCap/${name}_weights.pt
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

conda activate clip_prefix_caption

cd $CODE_DIR
python VALSE.py \
    --data_dir ${DATA_DIR} \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

conda deactivate 
