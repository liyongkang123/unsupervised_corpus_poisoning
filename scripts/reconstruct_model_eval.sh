#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list


model_list=("contriever-msmarco" "simlm-msmarco" "e5-base-v2" "tas-b" "retromae_msmarco_finetune" "dragon")

for sub_model in "${model_list[@]}"; do
    sbatch /home/yli8/unsupervised_corpus_poisoning/scripts/reconstruct_model_eval_sub.sh "${sub_model}"
done
