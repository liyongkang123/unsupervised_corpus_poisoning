#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.


# Activate conda
source ${HOME}/.bashrc
conda env list



nvidia-smi
attack_model_code_list=(  "simlm-msmarco"  )

dataset_list=(  "fiqa"  "webis-touche2020"  )  # "nq"
seed_list=(2024 2025 2026 )
attack_rate_list=("0.0001" "0.0005" "0.001" "0.005")
run_id_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
method_list=('random_noise' 'random_token' 'hotflip' 'unsupervised')

for sub_method in "${method_list[@]}"; do
for sub_attack_model_code in "${attack_model_code_list[@]}"; do
    for sub_dataset in "${dataset_list[@]}"; do
      for sub_seed in "${seed_list[@]}"; do
        for sub_attack_rate in "${attack_rate_list[@]}"; do
          for sub_run_id in "${run_id_list[@]}"; do
            sbatch scripts/corpus_poisoning_generate_sub.sh "${sub_method}" "${sub_attack_model_code}" "${sub_dataset}" "${sub_attack_rate}" "${sub_seed}" "${sub_run_id}"
        done
      done
done
done
done
done