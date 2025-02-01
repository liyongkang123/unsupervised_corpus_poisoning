'''
We do Corpus poisoning with  Random Noise, Random Token, HotFlip, and unsupervised ours.
1. Use the closest document to the cluster center found in the previous clustering step, and then attack, divided into id = 20 groups for attack
2. Use the generated documents for evaluation
'''

import json
import torch
import os
from transformers import AutoTokenizer, AutoModel, set_seed,BertForTokenClassification,set_seed,HfArgumentParser
from utils.load_data import load_beir_data
from models.model_reconstruction import BaseRecontructionModel
from models.model_attack import UnsupervisedAttack
from utils.utils import model_code_to_cmodel_name, move_to_cuda
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name, pooling_to_model_code, score_function_to_model_code,np_normalize
from utils.load_model import prefix_tokenizer
from utils.utils import format_passage,evaluate_recall,pooling_to_model_code,score_function_to_model_code,path_exist,mean_var
import torch.nn.functional as F
import copy


def split_list_into_20_parts(input_list):
    n = len(input_list)
    num_parts = 20  # Fixed into 20 parts
    # Initialize the result list, each part is initially an empty list
    result = [[] for _ in range(num_parts)]
    if n == 0:
        return result  # If the input list is empty, return 20 empty lists directly
    # Average length of each part
    avg_len = n // num_parts
    # The remainder determines the number of parts that need to be allocated one more element
    remainder = n % num_parts
    # Split index
    start = 0
    for i in range(num_parts):
        # If there is still a remainder, the current part is allocated one more element
        extra = 1 if i < remainder else 0
        end = start + avg_len + extra
        # Assign the corresponding slice to the current part
        result[i] = input_list[start:end]
        start = end  # Update the starting index
    return result

def main():
    device = torch.device( 'cuda' if torch.cuda.is_available() else "cpu" )

    attack_model_code_list = [ "simlm-msmarco" ]
    dataset_list = ["nq" ,"fiqa", "webis-touche2020" ] # "fiqa" "webis-touche2020" ]
    seed_list = [  2024, 2025, 2026 ] #2024, 2025, 2026
    attack_rate_list = [ "0.0001", "0.0005", "0.001", "0.005"  ] #"0.005"
    method_list = ['random_noise', 'random_token','hotflip', 'unsupervised' ]

    for method in method_list:
        for attack_mode_code in attack_model_code_list:
            for attack_dataset in dataset_list:
                for attack_rate in attack_rate_list:
                    result_all = []
                    for seed in seed_list:
                        set_seed(seed)
                        corpus, query, qrels = load_beir_data( attack_dataset, split='test')
                        if 'dpr' in attack_mode_code.lower():
                            raise NotImplementedError #
                        else:
                            query_encoder = AutoModel.from_pretrained(model_code_to_qmodel_name[attack_mode_code]).to(device)
                            ctx_encoder = AutoModel.from_pretrained(model_code_to_cmodel_name[attack_mode_code]).to(device)
                            q_tokenizer = AutoTokenizer.from_pretrained(model_code_to_cmodel_name[attack_mode_code])
                            ctx_tokenizer = q_tokenizer

                        query_encoder.eval()
                        query_encoder.to(device)
                        ctx_encoder.eval()
                        ctx_encoder.to(device)
                        q_tokenizer = prefix_tokenizer(attack_mode_code, tokenizer=q_tokenizer)
                        ctx_tokenizer = prefix_tokenizer(attack_mode_code, tokenizer=ctx_tokenizer)

                        decoder_base_model ="google-bert/bert-large-uncased"
                        load_model_path = f"output/recon_models/{attack_mode_code}/decoder_{(decoder_base_model).split('/')[-1]}/checkpoint-86350"
                        decoder = BertForTokenClassification.from_pretrained(load_model_path, num_labels=ctx_tokenizer.vocab_size)
                        decoder.eval()

                        recon_model = BaseRecontructionModel(context_encoder=copy.deepcopy(ctx_encoder), decoder=decoder).to(device)
                        recon_model.transfer_linear.load_state_dict( torch.load(f"{load_model_path}/transfer_linear.pt"))

                        attack_model = UnsupervisedAttack( model= recon_model, tokenizer= ctx_tokenizer , seed= seed)
                        attack_model.to(device)

                        pooling = next((value for key, value in pooling_to_model_code.items() if key in attack_mode_code.lower()), None)
                        score_function = next((value for key, value in score_function_to_model_code.items() if key in attack_mode_code.lower()), None)


                        cluster_target_file = f"output/results_corpus_attack/clustering/{attack_dataset}/{attack_mode_code}/cluster_target_do_rate_{attack_rate}_seed_{seed}.json"
                        with open(cluster_target_file, 'r') as f:
                            attack_target_do_dic = json.load(f)

                        attack_result_dic = {}

                        run_id_list = list(range(20))
                        for run_id in run_id_list:
                            attack_result_save_path = "output/results_corpus_attack/%s-generate/%s/%s/attack_rate_%s-run_id_%d-seed%d.json" % (
                                method, attack_dataset, attack_mode_code, attack_rate,
                                run_id, seed)

                            if os.path.exists(attack_result_save_path):
                                with open(attack_result_save_path, 'r') as f:
                                    new_data = json.load(f)

                                    attack_result_dic.update(new_data)
                            else:
                                print(f"File with runid {run_id} not found.")
                        # Calculating ASR
                        retrieval_result_file = f"output/retrieval_beir_results/{attack_dataset}/{attack_mode_code}/test_queries_qrels_only_relevant_score_beir.json"
                        with open(retrieval_result_file, 'r') as f:
                            retrieval_result_dic = json.load(f)

                        batch_size = 128

                        # Split data into batches
                        def batchify(data, batch_size):
                            for i in range(0, len(data), batch_size):
                                yield data[i:i + batch_size]

                        # Change the process;
                        # 1. Calculate the embedding of all adv_text first
                        adv_text_list = []
                        for index in attack_result_dic:
                            adv_text_list.append(attack_result_dic[index]['adv_text'])

                        # Get batches and process them
                        adv_text_embeddings = []
                        for batch in batchify(adv_text_list, batch_size):
                            # Tokenize and pad the batch
                            adv_encoded_input = ctx_tokenizer(batch, max_length=128 , padding=True, truncation=True, return_tensors="pt")
                            # get embeddings
                            with torch.no_grad(): # # Usually take the output of [CLS] token as sentence embedding
                                cls_embeddings =recon_model.encode_passage(move_to_cuda(adv_encoded_input))[:, 0]

                            adv_text_embeddings.append(cls_embeddings)
                        adv_text_embeddings = torch.cat(adv_text_embeddings, dim=0)  # [num_sentences, hidden_size]

                        # 2. Calculate the embedding of all queries
                        query_list = []
                        for q in query:
                            query_list.append(query[q])
                        query_embddings = []
                        for batch in batchify(query_list, batch_size):
                            query_encoded_input = ctx_tokenizer(batch, max_length=32, padding=True, truncation=True,
                                                      return_tensors="pt")
                            with torch.no_grad():
                                cls_embeddings =recon_model.encode_passage(move_to_cuda(query_encoded_input))[:, 0]
                            query_embddings.append(cls_embeddings)
                        query_embeddings = torch.cat(query_embddings, dim=0)  # [num_sentences, hidden_size]


                        # 3. Calculate the similarity
                        if score_function == 'cos_sim':
                            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                            adv_text_embeddings = F.normalize(adv_text_embeddings, p=2, dim=1)

                        elif score_function == 'dot':
                            pass
                        else:
                            raise ValueError(f"Unknown score function: {score_function}")
                        scores = torch.mm(query_embeddings, adv_text_embeddings.T)
                        # 4. Calculate the ASR

                        ASR = []
                        for idx ,q in  enumerate(query):
                            q_adv_score_max = max(scores[idx])
                            if q_adv_score_max > min(retrieval_result_dic[q].values()):
                                ASR.append(1)
                            else:
                                ASR.append(0)
                        asr_rate = sum(ASR) / len(ASR)
                        result_all.append(asr_rate)
                        print('ASR: ', asr_rate)
                    print("Attack method: ",method ,'Attack model: ', attack_mode_code ,'Attack dataset: ', attack_dataset, 'Attack rate: ', attack_rate, "ASR: ", sum(result_all)/len(result_all))

if __name__ == "__main__":
    main()