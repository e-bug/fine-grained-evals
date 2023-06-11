#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/ALBEF
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

name=albef_14m
task=VALSE
configs=configs_ours/${task}.yaml
ckpt=${CKPT_DIR}/ALBEF/ALBEF.pth
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
python VALSE.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

deactivate
