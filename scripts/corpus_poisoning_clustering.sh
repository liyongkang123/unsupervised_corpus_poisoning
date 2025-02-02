#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi
attack_model_code_list=(  "simlm-msmarco"  )

dataset_list=( "nq" "fiqa"  "webis-touche2020"  ) #
seed_list=(2024 2025 2026 )
attack_rate_list=("0.0001" "0.0005" "0.001" "0.005")

for sub_attack_model_code in "${attack_model_code_list[@]}"; do
    for sub_dataset in "${dataset_list[@]}"; do
      for sub_seed in "${seed_list[@]}"; do
        for sub_attack_rate in "${attack_rate_list[@]}"; do
            sbatch scripts/corpus_poisoning_clustering_sub.sh "${sub_attack_model_code}" "${sub_dataset}" "${sub_attack_rate}" "${sub_seed}"
        done
      done
done
done
