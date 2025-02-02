#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.



source ${HOME}/.bashrc
conda env list

nvidia-smi

dataset_list=( "trec_dl19"  "trec_dl20" "nq" "quora"  "fiqa" "webis-touche2020"  )  #['trec_dl19', "trec_dl20",   "nq",  "quora",  "fiqa", "webis-touche2020"]

method_list=('random_noise' 'random_token' 'hotflip' 'unsupervised')

for sub_method in "${method_list[@]}"; do
    for sub_dataset in "${dataset_list[@]}"; do
        sbatch /scripts/black_box_attack_sub.sh  "${sub_dataset}" "${sub_method}"
    done
done
