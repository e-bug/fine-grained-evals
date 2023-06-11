#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/BLIP2
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

ckpt=pretrain
task=Winoground
name=blip2_itm_${ckpt}
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/lavis/bin/activate

cd $CODE_DIR
python Winoground.py \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

deactivate 
