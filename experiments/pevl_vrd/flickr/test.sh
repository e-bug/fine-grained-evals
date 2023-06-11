#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/PEVL
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

name=pevl_vrd
task=Flickr
configs=configs_ours/${task}_512.yaml
ckpt=${CKPT_DIR}/PEVL/vrd.pth
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
python Retrieval.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --evaluate \
    # --device cpu

deactivate
