#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name="wino"
#SBATCH --output=test.out
#SBATCH --error=test.err

BASE_DIR=$HOME/scratch
CODE_DIR=${BASE_DIR}/projects/dm-fine/ALBEF
OUTS_DIR=${BASE_DIR}/outputs
ENVS_DIR=${BASE_DIR}/envs

module load gcc/9.3.0
module load python/3.8
module load cuda/11.2
# module load cudacore/.11.1.1
# module load nccl/2.8
module load cudnn/8.2.0
module load opencv/4.5.5
module load arrow/5.0.0

source $HOME/envs/kg-mml/bin/activate

for epoch in {00..29}; do

  name=albef_4m_ours_${epoch}
  task=Matching_winoground
  configs=configs_narval/${task}.yaml
  ckpt=$HOME/scratch/checkpoints/kg-mml/xpretrain-4m/ALBEF_repro/checkpoint_${epoch}.pth
  output=${OUTS_DIR}/${task}/${name}

  mkdir -p $output
  
  cd $CODE_DIR
  python Winoground.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --output_name ${name} \
    # --device cpu

done

deactivate 

