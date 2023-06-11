#!/bin/bash

BASE_DIR="/home/pmh864/projects/dm-fine"
CODE_DIR=${BASE_DIR}/ALBEF
OUTS_DIR=${BASE_DIR}/outputs
ENVS_DIR=${BASE_DIR}/envs

source ${HOME}/envs/albef/bin/activate

for epoch in {00..29}; do

  name=albef_4m_ours_${epoch}
  task=Matching_winoground
  configs=configs_hendrix/${task}.yaml
  ckpt=/projects/nlp/data/pmh864/checkpoints/dm-fine/ALBEF_repro/checkpoint_${epoch}.pth
  output=${OUTS_DIR}/${task}/${name}

  mkdir -p $output
  
  cd $CODE_DIR
  python Winoground.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    --device cpu

done

deactivate 

