#!/bin/bash

BASE_DIR="/mnt/disks/disk-1"
CODE_DIR=${BASE_DIR}/projects/ALBEF
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/outputs
ENVS_DIR=${BASE_DIR}/envs

name=albef_4m
task=RetrievalZS_flickr
configs=configs_dm/${task}.yaml
ckpt=${CKPT_DIR}/albef/ALBEF_4M.pth
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${HOME}/.bashrc
source activate ${ENVS_DIR}/albef

cd $CODE_DIR
python Retrieval.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --evaluate \
#    --device cpu

conda activate 

