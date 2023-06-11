#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/CLIP
CKPT_DIR=${BASE_DIR}/checkpoints
DATA_DIR=${BASE_DIR}/fine-grained-evals/data/svo_probes
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

name=clip_b32
task=SVO
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/clip/bin/activate

cd $CODE_DIR
python SVO.py \
    --data_dir ${DATA_DIR} \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

deactivate
