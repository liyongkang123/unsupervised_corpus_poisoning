#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

sub_eval_model_code=$1
sub_dataset=$2

python eval_beir_retrieval.py \
    --eval_model_code ${sub_eval_model_code} \
    --eval_dataset ${sub_dataset} \
    --per_gpu_eval_batch_size 128 \
    --split test
