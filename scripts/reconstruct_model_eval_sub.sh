#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.


# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

sub_model=$1

python reconstruction_model_eval.py \
    --recon_model_code ${sub_model} \
    --per_gpu_eval_batch_size 512