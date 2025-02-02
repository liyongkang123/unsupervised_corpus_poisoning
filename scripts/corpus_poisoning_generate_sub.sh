#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

sub_method=$1
sub_model=$2
sub_attack_dataset=$3
sub_attack_rate=$4
sub_seed=$5
sub_run_id=$6

python corpus_poisoning_generate.py \
    --method ${sub_method} \
    --recon_model_code ${sub_model} \
    --per_gpu_eval_batch_size 512 \
    --attack_dataset ${sub_attack_dataset} \
    --attack_rate  ${sub_attack_rate} \
    --seed ${sub_seed} \
    --run_id  ${sub_run_id}