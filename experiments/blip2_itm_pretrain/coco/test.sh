#!/bin/bash

BASE_DIR="/workdir"
CODE_DIR=${BASE_DIR}/fine-grained-evals/models/BLIP2
OUTS_DIR=${BASE_DIR}/fine-grained-evals/outputs
ENVS_DIR=${BASE_DIR}/envs
IMAGE_DIR=${BASE_DIR}/data/mscoco/
ANNOS_FN=${BASE_DIR}/data/mscoco/coco_test.json

ckpt=pretrain
task=COCO
name=blip2_${ckpt}
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

source ${ENVS_DIR}/lavis/bin/activate

cd $CODE_DIR
python Retrieval.py \
    --annotations_file ${ANNOS_FN} \
    --images_dir ${IMAGE_DIR} \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    --k_test 256 \
    # --device cpu

deactivate 
