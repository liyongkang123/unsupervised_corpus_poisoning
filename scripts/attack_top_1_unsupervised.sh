#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

attack_model_code_list=(  "simlm-msmarco" "e5-base-v2" )

dataset_list=( "trec_dl19"  "trec_dl20" "nq"   "nfcorpus" "hotpotqa"   "quora"   "scidocs"  "scifact"  "fiqa" "arguana" "webis-touche2020"  )


for sub_attack_model_code in "${attack_model_code_list[@]}"; do
    for sub_dataset in "${dataset_list[@]}"; do
        sbatch /home/yli8/unsupervised_corpus_poisoning/scripts/attack_top_1_unsupervised_sub.sh "${sub_attack_model_code}" "${sub_dataset}"
    done
done