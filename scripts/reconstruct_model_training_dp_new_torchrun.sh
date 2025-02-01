#!/bin/sh
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
source ${HOME}/.bashrc
conda env list


nvidia-smi

model_list=("contriever-msmarco" "simlm-msmarco" "e5-base-v2")
torchrun --standalone --nproc_per_node=4    --rdzv-backend=c10d    --rdzv-endpoint=localhost:2945687    reconstruction_model_train_trainer.py \
      --recon_model_code contriever-msmarco \
      --decoder_base_model google-bert/bert-large-uncased \
      --dataset_name msmarco \
      --dataset_split test \
      --per_device_train_batch_size 128 \
      --num_train_epochs 5 \
      --learning_rate 1e-5 \
      --seed 2024 \
      --save_steps 20000 \
      --logging_steps 100  \
      --passage_max_len 128