#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list

nvidia-smi

eval_model_code_list=("contriever" "contriever-msmarco" "dpr-single" "dpr-multi" "tas-b"
"dragon" "condenser" "simlm" "simlm-msmarco" "e5-base-v2"  "retromae_msmarco" "retromae_msmarco_finetune" "retromae_msmarco_distill")

dataset_list=( "msmarco"  "nq" "hotpotqa"   "dbpedia-entity"  "fever" "climate-fever" )

dataset_list=(  "trec-covid" "nfcorpus"   "fiqa" "arguana" "webis-touche2020" "cqadupstack" "quora"  "scidocs"  "scifact")

python dowloads_all_datasets.py

for sub_eval_model_code in "${eval_model_code_list[@]}"; do
    for sub_dataset in "${dataset_list[@]}"; do
        sbatch scripts/eval_beir_retrieval_sub.sh "${sub_eval_model_code}" "${sub_dataset}"
    done
done

