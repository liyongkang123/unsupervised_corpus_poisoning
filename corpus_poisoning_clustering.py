'''
We do Corpus poisoning with  Random Noise, Random Token, HotFlip, and unsupervised ours.
'''

import json
import torch
import os
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel, set_seed,BertForTokenClassification,set_seed,HfArgumentParser
from utils.load_data import load_beir_data
from utils.data_loader import BeirEncodeDataset
from models.model_reconstruction import BaseRecontructionModel
from models.model_attack import UnsupervisedAttack
from utils.utils import model_code_to_cmodel_name, move_to_cuda
from arguments_new import ModelArguments, DataArguments, TrainingArguments, AttackArguments
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name, pooling_to_model_code, score_function_to_model_code,np_normalize
from utils.load_model import prefix_tokenizer
import ir_measures
from ir_measures import *
from utils.utils import format_passage,evaluate_recall,pooling_to_model_code,score_function_to_model_code,path_exist,mean_var
import copy
import evaluate
import wandb
import time
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def main():
    device = torch.device( 'cuda' if torch.cuda.is_available() else "cpu" )

    wandb.init(  project="UCP_corpus_poisoning_clustering", )
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, attack_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        attack_args: AttackArguments

    wandb.config.update(vars(attack_args))
    wandb.config.update(vars(model_args))
    wandb.config.update(vars(data_args))
    wandb.config.update(vars(training_args))

    set_seed(training_args.seed)

    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    metric_wer = evaluate.load("wer")

    corpus, query, qrels = load_beir_data(attack_args.attack_dataset, split='test')  #
    attack_number_p = math.ceil(len(corpus) * attack_args.attack_rate)  # Round up, this is the number of documents that need to be generated during the attack
    print('The total number of clusters is： ',attack_number_p)
    corpus_ids = list(corpus.keys())

    file_name_dic={}
    for k_s in range(attack_number_p):
        file_name_dic[k_s]= "output/results_corpus_attack/%s-generate/%s/%s/k%d-s%d-seed%d.json" % (
         attack_args.method, attack_args.attack_dataset, attack_args.attack_mode_code, attack_number_p, k_s, training_args.seed)


    if 'dpr' in attack_args.attack_mode_code.lower():
        raise NotImplementedError
    else:
        query_encoder = AutoModel.from_pretrained(model_code_to_qmodel_name[attack_args.attack_mode_code]).to(device)
        ctx_encoder = AutoModel.from_pretrained(model_code_to_cmodel_name[attack_args.attack_mode_code]).to(device)
        q_tokenizer = AutoTokenizer.from_pretrained(model_code_to_cmodel_name[attack_args.attack_mode_code])
        ctx_tokenizer = q_tokenizer

    query_encoder.eval()
    query_encoder.to(device)
    ctx_encoder.eval()
    ctx_encoder.to(device)
    q_tokenizer = prefix_tokenizer(attack_args.attack_mode_code, tokenizer=q_tokenizer) # 增加对DistilBert的支持
    ctx_tokenizer = prefix_tokenizer(attack_args.attack_mode_code, tokenizer=ctx_tokenizer)

    load_model_path = f"output/recon_models/{attack_args.attack_mode_code}/decoder_{(model_args.decoder_base_model).split('/')[-1]}/checkpoint-86350"
    decoder = BertForTokenClassification.from_pretrained(load_model_path, num_labels=ctx_tokenizer.vocab_size) #-> 用base的模型
    decoder.eval()

    recon_model = BaseRecontructionModel(context_encoder=copy.deepcopy(ctx_encoder), decoder=decoder).to(device)
    recon_model.transfer_linear.load_state_dict( torch.load(f"{load_model_path}/transfer_linear.pt"))

    attack_model = UnsupervisedAttack( model= recon_model, tokenizer= ctx_tokenizer , seed= training_args.seed)
    attack_model.to(device)

    pooling = next((value for key, value in pooling_to_model_code.items() if key in attack_args.attack_mode_code.lower()), None)
    score_function = next((value for key, value in score_function_to_model_code.items() if key in attack_args.attack_mode_code.lower()), None)


    corpus_dataset = BeirEncodeDataset(corpus=corpus, tokenizer=ctx_tokenizer, max_length=data_args.passage_max_len)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=training_args.per_gpu_eval_batch_size, shuffle=False, # There is no need to shuffle here to avoid destroying the clustered index.
                                   num_workers=2, collate_fn=corpus_dataset.collate_fn)

    encoded = []
    for batch_idx, batch_data in enumerate(tqdm(corpus_dataloader)):
        batch_data = move_to_cuda(batch_data)
        with torch.no_grad():
            batch_corpus_embs =  ctx_encoder(**batch_data).last_hidden_state[:, 0, :]
            encoded.append(batch_corpus_embs.cpu().detach().numpy())
    c_embs = np.concatenate(encoded, axis=0)
    print("c_embs", c_embs.shape)
    #Let’s start clustering

    start_time = time.time()
    kmeans = faiss.Kmeans(c_embs.shape[1], attack_number_p, niter=300, verbose=True, gpu=True, seed= training_args.seed)
    kmeans.train(c_embs)
    centroids = kmeans.centroids
    print('The clustering time is： ',time.time()-start_time)


    index = faiss.IndexFlatL2 (c_embs.shape[1])
    index.add(c_embs)

    # Search for the nearest cluster center for each sample
    D, I = index.search(kmeans.centroids, 1)  # k=1 means only the nearest center is returned


    labels = I.flatten()  #

    # labels is the cluster label of each sample
    print(labels)
    return_data_dic={}

    attack_result_dic={}
    attack_result_p_ids=[]

    for index in range(attack_number_p):
        attack_result_dic[index] = {}

        attack_target_do = format_passage( text= corpus[ corpus_ids[I[index][0] ] ]['text'], title=corpus[ corpus_ids[I[index][0] ] ]['title']  )
        print('attack_target_document: ', attack_target_do)

        attack_result_dic[index]['target_document'] = attack_target_do
        attack_result_dic[index]['document_id'] = corpus_ids[I[index][0] ]

    cluster_target_file = f"output/results_corpus_attack/clustering/{attack_args.attack_dataset}/{attack_args.attack_mode_code}/cluster_target_do_rate_{attack_args.attack_rate}_seed_{training_args.seed}.json"
    path_exist(cluster_target_file)
    with open(cluster_target_file, 'w') as f:
        json.dump(attack_result_dic, f, indent=4)

if __name__ == "__main__":
    main()