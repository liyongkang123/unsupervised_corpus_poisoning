'''
Calculate the results of searching the entire corpus for the specified query and save the top-1000 results'''

import logging
import numpy as np
from transformers import (
    set_seed,
)
import wandb
logger = logging.getLogger(__name__)

from utils.logging import LoggingHandler
import config
from utils.load_data import load_beir_data
from transformers import AutoModel,AutoTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer,DPRQuestionEncoder,DPRQuestionEncoderTokenizer

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from models.beir_retrieval_model import RetrievalModel
import logging
import os
import json
import torch
import transformers
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name, pooling_to_model_code, score_function_to_model_code,np_normalize
from utils.load_model import prefix_tokenizer
def compress(results): # Keep top-2000
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]: #only keep top-2000
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

def main():
    args=  config.parse()
    print(args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="UCP_model_eval_beir",
        # track hyperparameters and run metadata
        config=vars(args),
    )

    set_seed(args.seed)
    args.result_output = 'output/retrieval_beir_results'

    result_output = f'{args.result_output}/{args.eval_dataset}/{args.eval_model_code}/{args.split}_queries_top1000_beir.json'
    # if os.path.isfile(result_output):
    #     exit()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #### load eval_dataset
    # split = 'test' or 'dev', the default is test
    corpus, queries, qrels = load_beir_data(args.eval_dataset,split=args.split)
    logging.info("Loading model...")
    pooling = next((value for key, value in pooling_to_model_code.items() if key in args.eval_model_code.lower()), None)
    score_function = next((value for key, value in score_function_to_model_code.items() if key in args.eval_model_code.lower()), None)
    if pooling is None or score_function is None:
        raise ValueError("Either pooling or score_function could not be determined from eval_model_code.")

    if 'dpr' in args.eval_model_code.lower():
        query_encoder= DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[args.eval_model_code]).to(device)
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_code_to_qmodel_name[args.eval_model_code])
        ctx_encoder = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[args.eval_model_code]).to(device)
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_code_to_cmodel_name[args.eval_model_code])
    else:
        query_encoder = AutoModel.from_pretrained(model_code_to_qmodel_name[args.eval_model_code]).to(device)
        ctx_encoder = AutoModel.from_pretrained(model_code_to_cmodel_name[args.eval_model_code]).to(device)
        q_tokenizer = AutoTokenizer.from_pretrained(model_code_to_cmodel_name[args.eval_model_code])
        ctx_tokenizer = q_tokenizer
    q_tokenizer = prefix_tokenizer(args.eval_model_code, tokenizer=q_tokenizer) #Add support for DistilBert
    ctx_tokenizer = prefix_tokenizer(args.eval_model_code, tokenizer=ctx_tokenizer)

    # The automodel model needs to be packaged into a DenseEncoderModel that can be used directly
    retrieval_model = RetrievalModel(query_encoder, ctx_model=ctx_encoder, q_tokenizer=q_tokenizer, ctx_tokenizer=ctx_tokenizer, max_seq_len=args.max_seq_length, max_query_len=args.max_query_length, pooling=pooling, normalize=False) #这里统一 normalize=False，因为如果要计算cosine similarity，我会在下一步进行
    model = DRES(retrieval_model, batch_size=args.per_gpu_eval_batch_size)


    logging.info(f"model: {model.model}")
    retriever = EvaluateRetrieval(model, score_function=score_function, k_values=[1, 5, 10, 50,100, 1000],) #  "cos_sim"  or "dot"  for dot-product
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print(ndcg, _map, recall, precision)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print('mrr: ',mrr)

    logging.info("Printing results to %s"%(result_output))
    sub_results = compress(results)
    output_dir_name = os.path.dirname(result_output)
    if not os.path.exists(output_dir_name ):
        os.makedirs(output_dir_name)
    with open(result_output, 'w') as f:
        json.dump(sub_results, f)
    print(f"Saved the top-1000 results to {result_output} is saved.")

    # Re-encode and record the results of all qrels
    # The reason for this is to prevent some relevant documents from not being in the top 1000.
    qrels_results_score = {}
    qrels_results_score_relevant_only = {}
    with torch.no_grad():
        for query_id in qrels:
            query = queries[query_id]
            query_encodings = retrieval_model.encode_queries([query],batch_size=1,show_progress=False)
            qrels_results_score[query_id] = {}
            qrels_results_score_relevant_only[query_id] = {}
            ctx_ids = qrels[query_id]
            ctxs = [corpus.get(c_id, {"title": "", "text": ""}) for c_id in ctx_ids]# # Prevent errors, return an empty string if it does not exist
            ctx_encodings = retrieval_model.encode_corpus(ctxs, batch_size=64, show_progress=False)  # Here is a list, and the elements in the list are dict
            if score_function == "cos_sim":
                query_encodings = np_normalize(query_encodings)
                ctx_encodings = np_normalize(ctx_encodings)
            elif score_function == "dot":
                pass
            scores = np.dot(query_encodings, ctx_encodings.T)
            scores = np.ravel(scores)
            for index, c_id in enumerate(qrels[query_id]):
                qrels_results_score[query_id][c_id] = scores[index].item()
                if qrels[query_id][c_id] > 0:
                    qrels_results_score_relevant_only[query_id][c_id] = scores[index].item()

    with open(f'{args.result_output}/{args.eval_dataset}/{args.eval_model_code}/{args.split}_queries_qrels_only_score_beir.json', 'w') as f:
        json.dump(qrels_results_score, f)
    print(f"Saved the  results to {args.result_output}/{args.eval_dataset}/{args.eval_model_code}/{args.split}_queries_qrels_only_score_beir.json is saved.")

    with open(f'{args.result_output}/{args.eval_dataset}/{args.eval_model_code}/{args.split}_queries_qrels_only_relevant_score_beir.json', 'w') as f:
        json.dump(qrels_results_score_relevant_only, f)
    print(f"Saved the  results to {args.result_output}/{args.eval_dataset}/{args.eval_model_code}/{args.split}_queries_qrels_only_relevant_score_beir.json is saved.")

if __name__ == '__main__':
    main()