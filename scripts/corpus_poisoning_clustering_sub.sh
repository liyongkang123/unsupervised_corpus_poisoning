#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.


# Activate conda
source ${HOME}/.bashrc
conda env list


nvidia-smi

sub_model=$1
sub_attack_dataset=$2
sub_attack_rate=$3
sub_seed=$4

python corpus_poisoning_clustering.py \
    --recon_model_code ${sub_model} \
    --per_gpu_eval_batch_size 512 \
    --attack_dataset ${sub_attack_dataset} \
    --attack_rate  ${sub_attack_rate} \
    --seed ${sub_seed}