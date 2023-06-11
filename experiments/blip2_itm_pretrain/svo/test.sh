#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/BLIP2
DATA_DIR=${BASE_DIR}/data/svo_probes
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs

ckpt=pretrain
task=SVO
name=blip2_itm_${ckpt}
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/lavis/bin/activate

cd $CODE_DIR
python SVO.py \
    --data_dir ${DATA_DIR} \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

deactivate 
