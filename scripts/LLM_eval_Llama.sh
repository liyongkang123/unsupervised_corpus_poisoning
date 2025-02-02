#!/bin/sh

#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

source ${HOME}/.bashrc
conda env list

nvidia-smi

python LLM_evalution_Llama.py