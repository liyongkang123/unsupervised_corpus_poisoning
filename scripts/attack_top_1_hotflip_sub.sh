#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

sub_attack_model_code=$1
sub_dataset=$2

python attack_top_1_hotflip.py \
    --method hotflip \
    --attack_mode_code ${sub_attack_model_code} \
    --attack_dataset ${sub_dataset} \
    --attack_split test \
    --attack_number 100 \
    --num_attack_epochs 3000 \
    --perplexity_model_name_or_path 'meta-llama/Llama-3.2-1B'