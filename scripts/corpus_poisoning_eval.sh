#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.


# Activate conda
source ${HOME}/.bashrc
conda env list

conda activate ir
cd /home/yli8/unsupervised_corpus_poisoning

nvidia-smi

python corpus_poisoning_eval.py