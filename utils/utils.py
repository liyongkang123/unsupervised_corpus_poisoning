import torch
from typing import Mapping, Dict, List
from transformers import set_seed
import random
from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import os

import numpy as np

def mean_var(values):
    # The input values is a list
    mean = np.mean(values)
    variance = np.var(values)
    return mean, variance

def np_normalize(embeddings, p=2, axis=1):
    norms = np.linalg.norm(embeddings, ord=p, axis=axis, keepdims=True)
    norms[norms == 0] = 1
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix}{query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix}{title.strip()} {text.strip()}'.strip()

def evaluate_recall(results, qrels, k_values=[10, 20, 50, 100, 500, 1000] ):
    cnt = {k: 0 for k in k_values}
    for q in results:
        sims = list(results[q].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        gt = qrels[q]
        found = 0
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
    recall = {}
    for k in k_values:
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)

    return recall

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def parse_qrels(qrels_path):
    qrels_dict = {}
    with open(qrels_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, rating = parts
                rating = int(rating)
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][docid] = rating

    return qrels_dict

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    # elif isinstance(model, RobustDenseModel):
    #     embeddings = model.encoder.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings # size is (30522,768)

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def move_to_device(sample, device='cuda'):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_device(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_device(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_device(sample)



def path_exist(file_name):
    folder_path = file_name if os.path.isdir(file_name) else os.path.dirname(file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory {folder_path} created.")
    else:
        print(f"Directory {folder_path} already exists.")

def path_change_between_windows_linux(path):
    if os.name == "nt":  # This is a Windows system.
        path = path.replace('/', '\\')
    elif os.name == "posix":  # This is a Linux or Unix-like system."
        path = path.replace('\\', '/')
    return path


def path_join(path_list):
    combined_path = path_change_between_windows_linux(os.path.join(*path_list))
    path_exist(combined_path)
    return combined_path

model_code_to_qmodel_name = {  # query encoder
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "tas-b": "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    "dragon": "facebook/dragon-plus-query-encoder",
    "condenser":"Luyu/co-condenser-marco", # coCondenser pre-trained on MS-MARCO collection
    "simlm": "intfloat/simlm-base-msmarco", #cosine similarity
    "simlm-msmarco": "intfloat/simlm-base-msmarco-finetuned", #cosine similarity
    "e5-base-v2": 'intfloat/e5-base-v2', #cosine similarity
    "cde": 'jxm/cde-small-v1', #cosine similarity
    'retromae_msmarco': 'Shitao/RetroMAE_MSMARCO', #Pre-trianed on the MSMARCO passage
    'retromae_msmarco_finetune': 'Shitao/RetroMAE_MSMARCO_finetune',
    'retromae_msmarco_distill': 'Shitao/RetroMAE_MSMARCO_distill',
}

model_code_to_cmodel_name = {  # ctx  encoder
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "tas-b":"sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    "dragon": "facebook/dragon-plus-context-encoder",
    "condenser":"Luyu/co-condenser-marco",
    "simlm": "intfloat/simlm-base-msmarco",
    "simlm-msmarco": "intfloat/simlm-base-msmarco-finetuned",
    "e5-base-v2": 'intfloat/e5-base-v2',
    "cde": 'jxm/cde-small-v1',
    'retromae_msmarco': 'Shitao/RetroMAE_MSMARCO',
    'retromae_msmarco_finetune': 'Shitao/RetroMAE_MSMARCO_finetune',
    'retromae_msmarco_distill': 'Shitao/RetroMAE_MSMARCO_distill',
}

prefix_to_model_code = {
    'cde':{'query_prefix':"search_query: ", "document_prefix": "search_document: "},
    'e5-base-v2':{'query_prefix':"query: ", "document_prefix": "passage: "}, #no need to add prefix for instruct models, however, we need to add prefix for e5 base
}

pooling_to_model_code = {
    "bert": "cls",
    "contriever": "mean",
    'dpr': "mean",
    "ance":'cls',
    "tas-b": 'cls',
    "dragon": 'cls',
    "condenser": 'cls',
    "simlm": 'cls',
    "retromae": 'cls',
    "e5": 'mean',
}
score_function_to_model_code = {
    'contriever': 'dot',
    'dpr': 'dot',
    "ance": 'dot',
    "tas-b": 'dot',
    "dragon": 'dot',
    "condenser": 'dot',
    'simlm': 'cos_sim',
    'cde': 'cos_sim',
    'retromae': 'dot',
    'e5': 'cos_sim',
}