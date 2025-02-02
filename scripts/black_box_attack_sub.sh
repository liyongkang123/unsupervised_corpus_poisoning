#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.



source ${HOME}/.bashrc
conda env list

nvidia-smi
sub_attack_dataset=$1
sub_method=$2

python black_box_attack.py \
    --attack_dataset ${sub_attack_dataset} \
    --method ${sub_method}