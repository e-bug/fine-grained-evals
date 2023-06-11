#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/ALBEF
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs
DATA_DIR=${BASE_DIR}/data/vsr

name=albef_4m
task=VSR_random
configs=configs_ours/${task}.yaml
ckpt=${CKPT_DIR}/ALBEF/ALBEF_4M.pth
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
python VSR.py \
    --data_dir ${DATA_DIR} \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
#    --device cpu

deactivate
