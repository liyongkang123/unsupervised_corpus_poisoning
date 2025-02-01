from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
# import json
import ujson as json # 使用ujson代替json，速度更快
import os
import logging
import csv
from itertools import islice
from torch.utils.data import Dataset, DataLoader
import torch
import mmap

logger = logging.getLogger(__name__)


class BeirGenericDataLoader: # special for BEAR datasets

    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl",
                 query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        with open(self.corpus_file, encoding='utf8') as fIn:
            lines = fIn.readlines()
            total_lines = len(lines)

            for line in tqdm(lines, total=total_lines):
                line = json.loads(line)
                self.corpus[line["_id"]] = {
                    "text": line["text"],
                    "title": line["title"],
                }

    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):

        open_file = open(self.qrels_file, encoding="utf-8")

        reader = csv.reader(open_file,delimiter="\t", quoting=csv.QUOTE_MINIMAL)

        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

        open_file.close()

class BeirEncodeDataset(Dataset):
    def __init__(self, text=None,*,corpus=None, query=None, tokenizer=None, max_length=None):
        if corpus:
            self.corpus_texts = ['{} {}'.format(corpus[doc].get('title', ''), corpus[doc]['text']).strip() for doc in corpus]
            self.id = [doc for doc in corpus]
            self.encode_passage = True
            self.encode_query = False
        elif query:
            self.corpus_texts = list(query.values())
            self.id = list(query.keys())
            self.encode_passage = False
            self.encode_query = True
        else:
            self.corpus_texts = text
            self.id = None
            self.encode_passage = False
            self.encode_query = False
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus_texts)

    def __getitem__(self, idx):
        text = self.corpus_texts[idx]
        return text


    def collate_fn(self, batch):
        tokenized_text = self.tokenizer(batch, encode_query=self.encode_query, encode_passage=self.encode_passage, padding="max_length", truncation=True, max_length=self.max_length,
                                        return_tensors="pt")
        return {k: v for k, v in tokenized_text.items()}



from typing import List, Tuple, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
@dataclass
class TrainCollator:
    encode_query: bool
    encode_passage: bool
    max_length: int
    tokenizer: PreTrainedTokenizer
    def __call__(self, features: List[str]):
        tokenized_text = self.tokenizer(features, encode_query=self.encode_query, encode_passage=self.encode_passage, padding="max_length", truncation=True, max_length=self.max_length,
                                        return_tensors="pt")
        return {k: v for k, v in tokenized_text.items()}


class Attack_Batch_Dataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float().to('cuda')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]