import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import faiss
import torch
import logging
import os
import requests
import zipfile
from tqdm import tqdm
from data_loader import BeirGenericDataLoader #beir.datasets
import random
import numpy as np
from torch.utils.data import DataLoader
import time
from collections import Counter
from transformers import default_data_collator
from sklearn.cluster import KMeans
from data_loader import Attack_Batch_Dataset
from datasets import load_dataset
from utils.utils import parse_qrels

logger = logging.getLogger(__name__)

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()

def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))

def load_beir_data(dataset_name, split):  #from beir, only for test
    ## The reason why this function is called first is that the trec_dl19 and trec_dl20 datasets need to call the load_beir_data() function repeatedly
    if '-train' in dataset_name:  # Because there is a dataset named nq-train
        split = 'train'
    elif split == 'test' and dataset_name in ['msmarco', 'trec_dl19', 'trec_dl20']:  # Because the msmarco dataset does not use test but dev
        split = 'dev'

    # Load datasets
    if dataset_name=='trec_dl19' or dataset_name=='trec_dl20':
        print('loading trec data ing...')
        if dataset_name =='trec_dl19':
            dataset_name = "crystina-z/msmarco-passage-dl19"
            trec_hf_data = load_dataset(dataset_name, token=True, split=split,trust_remote_code=True)  # dl19 is only used for testing
            qrels_path = 'datasets/trec_dl19/2019qrels-pass.txt'
        elif dataset_name =='trec_dl20':
            dataset_name = "crystina-z/msmarco-passage-dl20"
            trec_hf_data = load_dataset(dataset_name, token=True, split=split,trust_remote_code=True)
            qrels_path = 'datasets/trec_dl20/2020qrels-pass.txt'
        corpus, _,_= load_beir_data("msmarco", split='dev')
        qrels = parse_qrels(qrels_path)
        # Build a dictionary, making sure row['query_id'] is in the keys of qrels
        queries = {row['query_id']: row['query'] for row in trec_hf_data if row['query_id'] in qrels.keys()}

    else:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = os.path.join(out_dir, dataset_name)
        if not os.path.exists(data_path):
            data_path = download_and_unzip(url, out_dir)
        print(data_path)
        data = BeirGenericDataLoader(data_path)
        corpus, queries, qrels = data.load(split=split) # # corpus is a dict, containing all passage information, including title, text, id, etc.

    return corpus, queries, qrels


def tokenization(examples, tokenizer, max_seq_length, pad_to_max_length):
    q_feat = tokenizer(examples["sent0"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)
    c_feat = tokenizer(examples["sent1"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)

    ret = {}
    for key in q_feat:
        ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]
    return ret

def create_batches(data, batch_size):
    #Split the data by batch_size and calculate the average value of each batch.
    if len(data) <= batch_size:
        return [data]

    num_batches = len(data) // batch_size
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    if len(data) % batch_size != 0:
        batches.append(data[num_batches * batch_size:])
    return batches

def batch_average(batches):
    avg_data = np.array([np.mean(batch, axis=0) for batch in batches])
    return avg_data