# unsupervised_corpus_poisoning_public
This is the code repository for our SIGIR 2025 Full paper [《Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval
》](https://arxiv.org/abs/2504.17884)..

## Datasets

The datasets used in the experiments are from the BEIR library. The datasets are stored in the datasets/ folder. These datasets will download automatically when you run the code.

## Requirements
- Python ,PyTorch , numpy, pandas, beir,transformers, sentence_transformers, sklearn, wandb, openai, 
- ir_measures, faiss, evaluate,logging, 
- If you do not want to use wandb, you can comment out all code with *wandb* in the code.
- You need to install the *beir* library(https://github.com/beir-cellar/beir), and the *colbert* library(https://github.com/stanford-futuredata/ColBERT)

# Steps to reproduce the main results
## Train the Reconstruction Model
- 1, Run `python dowloads_all_datasets.py` to dowload all datasets we need. The datasets are saved in `datasets/`.
- 2, Run `sbatch scripts/reconstruct_model_training_dp_new_torchrun.sh.sh` to train the reconstruct_model. The models are trained by msmarco and  saved in `output/recon_models`.
- 3, Run `sbatch scripts/reconstruct_model_eval.sh` to evaluate the reconstruction model on the corpus of NQ. 

## Top-1 Attack 
### White box
- 1, Run `sbatch scripts/eval_beir_retrieval.sh` to evaluate the retrieval performance of different retrieval model on Beir datasets. The retrieval results are saved in `output/retrieval_beir_results`.
- 2, Run `sbatch scripts/attack_top_1_unsupervised.sh` to generate the adversarial examples of top-1 attack by our method. The adversarial examples are saved in `output/attack_results/unsupervised`.
- 3, Run `sbatch scripts/attack_top_1_hotflip.sh` to generate the adversarial examples of top-1 attack by HotFlip. The adversarial examples are saved in `output/attack_results/hotflip`.
- 4, Run `sbatch scripts/attack_top_1_random_noise.sh` to generate the adversarial examples of top-1 attack by Random Noise. The adversarial examples are saved in `output/attack_results/random_noise`.
- 5, Run `sbatch scripts/attack_top_1_random_token.sh` to generate the adversarial examples of top-1 attack by Random Token. The adversarial examples are saved in `output/attack_results/random_token`.
### Black-box 
- 1, We use the same generated results by the above white box attack.
- 2, Run `sbatch scripts/black_box_attack.sh` to transfer attacks to other retrieval models.


## Corpus Poisoning
- 1, Run `sbatch scripts/corpus_poisoning_clustering.sh` to get the centroids and the nearest documents of all clusters.
- 2, Run `sbatch scripts/corpus_poisoning_generate.sh` to generate the adversarial examples of corpus poisoning by all four methods.
- 3, Run `sbatch scripts/corpus_poisoning_eval.sh` to evaluate the results.


## Note
The above steps omit the process of switching random seeds and conducting the experiment three times.  

In the explicitly mentioned parts of the paper, we conducted three experiments using seed values of 2024, 2025, and 2026, and averaged their results.

## Thanks 
- We thank the authors of the 《Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval 》and 《Poisoning Retrieval Corpora by Injecting Adversarial Passages》 for their code and data. We use their code to reproduce the results of the HotFlip attack.

# Citation
If you find this code useful, I would greatly appreciate it if you could cite our paper:
```
@inproceedings{li2025unsupervised_corpus_poisoning,
  title={Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval},
  author={Yongkang Li and Panagiotis Eustratiadis and Simon Lupart and Evangelos Kanoulas},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, {SIGIR} 2025},
  year={2025},
  url = {https://doi.org/10.48550/arXiv.2504.17884},
  doi = {10.48550/ARXIV.2504.17884},
}
```