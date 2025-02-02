'''
Attack the top-1 document in the retrieval results
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
from utils.load_model import prefix_tokenizer
from arguments_new import ModelArguments, DataArguments, TrainingArguments, AttackArguments
import random
import config
import ir_measures
from ir_measures import *
from utils.utils import format_passage,evaluate_recall,pooling_to_model_code,score_function_to_model_code,path_exist,mean_var
import torch.optim as optim
import torch.nn.functional as F
import heapq
import copy
from utils.utils import GradientStorage
import time
import logging
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
import evaluate
from textstat import textstat
import gc
from scipy import stats
from attack_top_1_unsupervised import embedding_score_two
import wandb


def main():

    wandb.init(  project="UCP_attack_generate_random_noise", )
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

    wandb.config.update(vars(attack_args))  #
    wandb.config.update(vars(model_args))  #
    wandb.config.update(vars(data_args))  #
    wandb.config.update(vars(training_args))  #

    set_seed(training_args.seed)
    assert attack_args.method == "random_noise"

    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    metric_wer = evaluate.load("wer")

    corpus, query, qrels = load_beir_data(attack_args.attack_dataset, split='test')  # 'test' 我们用corpus进行训练


    query_sample = random.sample(list(query.keys()), min(len(query), attack_args.attack_number))
    print(f"Attack {len(query_sample)} queries")

    #
    retrieval_results_path = f"output/retrieval_beir_results/{attack_args.attack_dataset}/{attack_args.attack_mode_code}"
    with open(f"{retrieval_results_path}/{attack_args.attack_split}_queries_top1000_beir.json", 'r') as f:
        retrieval_results_queries_top1000 = json.load(f)
    with open(f"{retrieval_results_path}/{attack_args.attack_split}_queries_qrels_only_score_beir.json", 'r') as f:
        retrieval_results_queries_qrels_only = json.load(f)
    with open(f"{retrieval_results_path}/{attack_args.attack_split}_queries_qrels_only_relevant_score_beir.json", 'r') as f: #_queries_qrels_only_relevant_score_beir
        retrieval_results_queries_qrels_only_relevant_score = json.load(f)


    rrqt_sample,rrqqo_sample, qrels_sample,rrqqor_sample ={},{},{},{}
    for q in query_sample:
        rrqt_sample[q]=retrieval_results_queries_top1000[q]
        qrels_sample[q]=qrels[q]
        rrqqo_sample[q]=retrieval_results_queries_qrels_only[q]
        rrqqor_sample[q] = retrieval_results_queries_qrels_only_relevant_score[q]
    retrieval_results_sample = ir_measures.calc_aggregate([nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10, nDCG @ 3, nDCG @ 20, nDCG @ 50],
                                        qrels_sample, rrqt_sample)
    print(retrieval_results_sample)


    # Load the model
    recon_model_c_path = model_code_to_cmodel_name.get(attack_args.attack_mode_code, attack_args.attack_mode_code)
    ctx_encoder = AutoModel.from_pretrained(recon_model_c_path)
    tokenizer = AutoTokenizer.from_pretrained(recon_model_c_path) #
    tokenizer = prefix_tokenizer(attack_args.attack_mode_code, tokenizer=tokenizer)

    ctx_encoder.eval()
    ctx_encoder.to(device)

    load_model_path = f"output/recon_models/{attack_args.attack_mode_code}/decoder_{(model_args.decoder_base_model).split('/')[-1]}/checkpoint-86350"
    decoder = BertForTokenClassification.from_pretrained(load_model_path, num_labels=tokenizer.vocab_size) #-> 用base的模型
    decoder.eval()

    recon_model = BaseRecontructionModel(context_encoder=copy.deepcopy(ctx_encoder), decoder=decoder).to(device)
    recon_model.transfer_linear.load_state_dict(torch.load(f"{load_model_path}/transfer_linear.pt"))

    pooling = next((value for key, value in pooling_to_model_code.items() if key in attack_args.attack_mode_code.lower()), None)
    score_function = next((value for key, value in score_function_to_model_code.items() if key in attack_args.attack_mode_code.lower()), None)

    appeared_10, appeared_20, appeared_50, appeared_100 = [], [], [], []
    attack_results = {}
    ndcg_10_before, ndcg_10_after = [], []  #
    asr_all, bleu_all, wer_all = [], [], []
    adv_text_all, top_1_targets_documents_all = [], []
    #
    start_time = time.time()
    for idx, q in enumerate(query_sample):
        set_seed(training_args.seed + idx)  #

        attack_results[q]={} #
        qrels_q = {q: qrels[q]}  #
        rrqt_q = {q: retrieval_results_queries_top1000[q]}
        rrqqo_q = {q: retrieval_results_queries_qrels_only[q]}

        rrqt_q_adv=copy.deepcopy(rrqt_q)

        re_raw = ir_measures.calc_aggregate([nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10, nDCG @ 3, nDCG @ 20, nDCG @ 50], qrels_q, rrqt_q)

        print(re_raw)

        re_raw_dic = {str(key): value for key, value in re_raw.items()}
        ndcg_10_before.append(re_raw_dic['nDCG@10'])

        rrqt_q_top1 = sorted(retrieval_results_queries_top1000[q].items(), key=lambda item: item[1], reverse=True)[0]
        print(f"rrqt_q_top1:{rrqt_q_top1}")

        top_1_text = corpus[rrqt_q_top1[0]] #
        top_1_text = format_passage(top_1_text['text'],top_1_text['title'] )
        print(top_1_text)
        attack_results[q]['top_1_target_text'] = top_1_text
        attack_results[q]['query_text'] = query[q]
        # attack_results[q]['retrieval_result']= rrqt_q_top1
        attack_results[q]['rank_metrics_before'] = re_raw_dic
        #
        query_input = tokenizer(query[q], return_tensors='pt', padding=True, truncation=True, encode_query=True,
                              max_length=data_args.query_max_len).to(device)

        # random noise  attack
        target_d_inputs = tokenizer(top_1_text, truncation=True, max_length=data_args.passage_max_len, return_tensors="pt", encode_passage=True).to(device)
        top_1_text_truncated= tokenizer.decode(target_d_inputs['input_ids'][0], skip_special_tokens=True)
        top_1_targets_documents_all.append(top_1_text_truncated) #
        mean = 0
        std = 1
        beta = attack_args.random_noise_rate
        with torch.no_grad():
            target_d_emb = recon_model.encode_passage(target_d_inputs)
            gaussian_noise = torch.normal(mean, std, size=target_d_emb.shape)  #
            encode_embedding_noise = target_d_emb + gaussian_noise.to(device) * beta

            # decoder
            encode_embedding_noise = recon_model.transfer_linear(encode_embedding_noise)
            decoder_outputs = recon_model.context_decoder(
                inputs_embeds=encode_embedding_noise,
                labels=target_d_inputs['input_ids'],  #
            )
            cross_entropy_loss_adv = decoder_outputs['loss']
            decoder_outputs_log = decoder_outputs['logits']  #

            #
            adv_text_index = torch.argmax(decoder_outputs_log, dim=-1)
            adv_text_list = tokenizer.batch_decode(adv_text_index, skip_special_tokens=False)
            adv_text = adv_text_list[0]

        adv_text_inputs = tokenizer(adv_text, return_tensors='pt', padding=True, truncation=True, encode_passage=True,
                                    max_length=data_args.passage_max_len)
        q_adv_score = embedding_score_two(recon_model, query_input, adv_text_inputs,
                                          score_function=score_function)
        d_adv_score = embedding_score_two(recon_model, target_d_inputs, adv_text_inputs,
                                          score_function=score_function)
        adv_list=[]
        adv_list.append({
            'epoch': 0,
            'q_adv_score': q_adv_score,
            'd_adv_score': d_adv_score,
            'adv_text': adv_text_list[0],
        })
        # Sort the list by 'd_adv_score' in descending order
        adv_list = sorted(adv_list, key=lambda x: x['d_adv_score'], reverse=True)
        print(adv_list[:10])
        adv_list_2 = sorted(adv_list, key=lambda x: x['q_adv_score'], reverse=True)
        print(adv_list_2[:10])

        #
        rrqt_q_adv[q]['adv_d'] = adv_list[0]['q_adv_score']
        re_adv = ir_measures.calc_aggregate(
            [nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10, nDCG @ 3, nDCG @ 20, nDCG @ 50], qrels_q, rrqt_q_adv)
        re_adv_dic = {str(key): value for key, value in re_adv.items()}
        ndcg_10_after.append(re_adv_dic['nDCG@10'])
        adv_qrels = {q: {"adv_d": 1 }}
        adv_eval = evaluate_recall(rrqt_q_adv, adv_qrels) #


        attack_results[q]['rank_metrics_after'] = re_adv_dic
        attack_results[q]["adv_text"] = adv_list[0]['adv_text']
        adv_text_all.append(adv_list[0]['adv_text'])
        bleu_score = metric_bleu.compute(predictions=[adv_list[0]['adv_text']], references=[top_1_text_truncated],
                                         lowercase=True)['score']  #
        bleu_all.append(bleu_score)
        wer_score = metric_wer.compute(predictions=[adv_list[0]['adv_text']], references=[top_1_text_truncated])
        wer_all.append(wer_score)

        attack_results[q]['adv_text_bleu_score'] = bleu_score
        attack_results[q]['adv_text_wer_score'] = wer_score
        attack_results[q]["adv_q_dot"] = adv_list[0]['q_adv_score']
        attack_results[q]["adv_d_dot"] = adv_list[0]['d_adv_score']
        attack_results[q]["adv_list"] = adv_list[:10]
        attack_results[q]["adv_list2"] = adv_list_2[:10]
        attack_results[q]["adv_eval"] = adv_eval

        appeared_10.append(adv_eval['Recall@10'])
        appeared_20.append(adv_eval['Recall@20'])
        appeared_50.append(adv_eval['Recall@50'])
        appeared_100.append(adv_eval['Recall@100'])

        #
        if adv_list[0]['q_adv_score'] > min(rrqqor_sample[q].values()):
            asr_all.append(1)
            attack_results[q]["asr_score"] = 1
        else:
            asr_all.append(0)
            attack_results[q]["asr_score"] = 0

        print('query attack finished', 'query id: ', q, 'adv_eval:', adv_eval)
    end_time = time.time()
    print(f"Attack finished in {end_time - start_time} seconds, the average time for each query is {(end_time - start_time) / len(query_sample)} seconds.")
    #  attack_results
    attack_results_path = (f"output/attack_results/{attack_args.method}/{attack_args.attack_dataset}/"
                           f"{attack_args.attack_mode_code}/{attack_args.attack_split}_top1_attack_lr-{training_args.learning_rate}_lm_loss_clip_max-{attack_args.lm_loss_clip_max}_seed-{training_args.seed}"
                           f"_noise_rate-{attack_args.random_noise_rate}.json")
    path_exist(attack_results_path)
    with open(attack_results_path, 'w') as f:
        json.dump(attack_results, f, indent=4)

    #  recall
    appeared_10_score,_ = mean_var(appeared_10)
    appeared_20_score,_ = mean_var(appeared_20)
    appeared_50_score,_ = mean_var(appeared_50)
    appeared_100_score,_ = mean_var(appeared_100)
    print(f"Appeared@10: {appeared_10_score}, Appeared@20: {appeared_20_score}, Appeared@50: {appeared_50_score}, Appeared@100: {appeared_100_score}")

    #
    ndcg_10_before_score, _ = mean_var(ndcg_10_before)
    ndcg_10_after_score, _ = mean_var(ndcg_10_after)
    print(f"nDCG@10 before: {ndcg_10_before_score}, nDCG@10 after: {ndcg_10_after_score}")
    #
    t_stat, p_value = stats.ttest_rel(ndcg_10_before, ndcg_10_after) #
    print("t-statistic:", t_stat)
    print("p-value:", p_value)
    #
    alpha = 0.05
    #
    if p_value < alpha:
        print("The null hypothesis is rejected and the result is statistically significant.")
    else:
        print("The null hypothesis cannot be rejected and the results are not statistically significant.")

    # ASR
    asr_score,_ = mean_var(asr_all)
    print(f"ASR: {asr_score}")

    #
    bleu_score_mean, bleu_score_var =mean_var(bleu_all)
    wer_score_mean, wer_score_var = mean_var(wer_all)
    print(f"BLEU mean: {bleu_score_mean}, BLEU var: {bleu_score_var}, WER mean: {wer_score_mean}, WER var: {wer_score_var}")

    # PPL
    from evaluate import load
    perplexity = load("perplexity", module_type="metric")
    adv_text_perplexity_results = perplexity.compute(predictions=adv_text_all, model_id=attack_args.perplexity_model_name_or_path)
    print("mean_perplexity of adv text: ", adv_text_perplexity_results["mean_perplexity"])
    # #
    del perplexity
    gc.collect()  #
    torch.cuda.empty_cache()
    perplexity = load("perplexity", module_type="metric")
    top_1_targets_documents_text_perplexity_results = perplexity.compute(predictions=top_1_targets_documents_all, model_id=attack_args.perplexity_model_name_or_path)
    print("mean_perplexity of top_1_targets_documents_text: ", top_1_targets_documents_text_perplexity_results["mean_perplexity"])

    # Readability
    readability_adv_text = [textstat.dale_chall_readability_score(text) for text in adv_text_all]
    readability_top_1_targets_documents = [textstat.dale_chall_readability_score(text) for text in top_1_targets_documents_all]
    readability_adv_text_mean, readability_adv_text_var = mean_var(readability_adv_text)
    readability_top_1_targets_documents_mean, readability_top_1_targets_documents_var = mean_var(readability_top_1_targets_documents)
    print(f"Readability of adv text: mean: {readability_adv_text_mean}, var: {readability_adv_text_var}")
    print(f"Readability of top_1_targets_documents: mean: {readability_top_1_targets_documents_mean}, var: {readability_top_1_targets_documents_var}")

    #
    # import language_tool_python
    # grammar_tool = language_tool_python.LanguageToolPublicAPI('en-US')
    # grammar_errors_adv_text = [len(grammar_tool.check(text)) for text in adv_text_all]
    # grammar_errors_top_1_targets_documents = [len(grammar_tool.check(text)) for text in top_1_targets_documents_all]
    # grammar_errors_adv_text_mean, grammar_errors_adv_text_var = mean_var(grammar_errors_adv_text)
    # grammar_errors_top_1_targets_documents_mean, grammar_errors_top_1_targets_documents_var = mean_var(grammar_errors_top_1_targets_documents)
    # print(f"Grammar errors of adv text: mean: {grammar_errors_adv_text_mean}, var: {grammar_errors_adv_text_var}")
    # print(f"Grammar errors of top_1_targets_documents: mean: {grammar_errors_top_1_targets_documents_mean}, var: {grammar_errors_top_1_targets_documents_var}")


if __name__ == '__main__':
    main()