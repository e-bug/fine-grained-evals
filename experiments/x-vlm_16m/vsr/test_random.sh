#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/X-VLM
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

name=x-vlm_16m
task=VSR_random
configs=configs_ours/${task}.yaml
ckpt=${CKPT_DIR}/X-VLM/16m_base_model_state_step_199999.th
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/x-vlm/bin/activate

cd $CODE_DIR
python VSR.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

deactivate 
