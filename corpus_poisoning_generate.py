'''
We do Corpus poisoning with  Random Noise, Random Token, HotFlip, and unsupervised ours.
Use the closest document to the cluster center found in the previous clustering step, and then attack, dividing it into id = 20 groups for attack'''

import json
import torch
import os
from transformers import AutoTokenizer, AutoModel, set_seed,BertForTokenClassification,set_seed,HfArgumentParser
from models.model_reconstruction import BaseRecontructionModel
from models.model_attack import UnsupervisedAttack
from arguments_new import ModelArguments, DataArguments, TrainingArguments, AttackArguments
from utils.utils import model_code_to_qmodel_name, model_code_to_cmodel_name, pooling_to_model_code, score_function_to_model_code,np_normalize
from utils.load_model import prefix_tokenizer
from ir_measures import *
from utils.utils import format_passage,evaluate_recall,pooling_to_model_code,score_function_to_model_code,path_exist,mean_var
import torch.optim as optim
import copy
import evaluate
import wandb
from attack_top_1_unsupervised import embedding_score_two
from attack_top_1_hotflip import evaluate_acc_batch,get_emb,hotflip_candidate,hotflip_candidate_score
from utils.utils import GradientStorage

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

    wandb.init(  project="UCP_corpus_poisoning_generate", )
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
    q_tokenizer = prefix_tokenizer(attack_args.attack_mode_code, tokenizer=q_tokenizer)
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



    attack_result_dic={}
    attack_result_p_ids=[]

    cluster_target_file = f"output/results_corpus_attack/clustering/{attack_args.attack_dataset}/{attack_args.attack_mode_code}/cluster_target_do_rate_{attack_args.attack_rate}_seed_{training_args.seed}.json"

    with open(cluster_target_file, 'r') as f:
        attack_target_do_dic = json.load(f)

    attack_number_p = len(attack_target_do_dic)
    attack_id = attack_target_do_dic.keys()
    runs = split_list_into_20_parts(list(attack_id))

    # Find the current run_id
    attack_sub_ids = runs[attack_args.run_id]
    print('attack_sub_ids: ', attack_sub_ids)
    if len(attack_sub_ids) == 0:
        pass
    else:
        for index in attack_sub_ids:
            attack_result_dic[index] = {}

            attack_target_do = attack_target_do_dic[str(index)]["target_document"]
            print('attack_target_document: ', attack_target_do)

            attack_result_dic[index]['attack_cluster_id'] = index
            attack_result_dic[index]['target_document'] = attack_target_do
            attack_result_dic[index]['document_id'] = attack_target_do_dic[str(index)]["document_id"]

            target_d_inputs = ctx_tokenizer(attack_target_do, truncation=True, max_length=data_args.passage_max_len,
                                            return_tensors="pt", encode_passage=True).to(device)

            # 2. Generate adversarial documents. Here are 4 methods
            if attack_args.method == 'unsupervised':
                attack_model._initialize_weights()
                optimizer = optim.AdamW(attack_model.noise_model.parameters(), lr=training_args.learning_rate,
                                        weight_decay=training_args.weight_decay)  # Adjust lr as needed
                adv_list = []
                for epoch in range(attack_args.num_attack_epochs):
                    # print(f"Epoch {epoch}")
                    optimizer.zero_grad(set_to_none=False)  # Clear the gradients before backward pass
                    outputs = attack_model(target_d_inputs)
                    mse_loss, cross_entropy_loss = outputs['mse_loss_noise'], outputs['cross_entropy_loss_adv']
                    loss = mse_loss - cross_entropy_loss.clamp(max=attack_args.lm_loss_clip_max)
                    loss.backward()
                    optimizer.step()

                    adv_text_inputs = ctx_tokenizer(outputs['adv_text'], return_tensors='pt', padding=True, truncation=True,
                                                max_length=data_args.passage_max_len, encode_passage=True)
                    d_adv_score = embedding_score_two(recon_model, target_d_inputs, adv_text_inputs,
                                                      score_function=score_function)
                    adv_list.append({
                        'epoch': epoch,
                        'd_adv_score': d_adv_score,
                        'adv_text': outputs['adv_text'],
                        'cross_entropy_loss': cross_entropy_loss.item(),
                        'mse_loss': mse_loss.item()
                    })
                # Sort the list by 'd_adv_score' in descending order
                adv_list = sorted(adv_list, key=lambda x: x['d_adv_score'], reverse=True)
                adv_text = adv_list[0]['adv_text']
                print('adv_text: ', adv_text)

            elif attack_args.method == 'hotflip':
                embeddings = ctx_encoder.embeddings.word_embeddings
                embedding_gradient = GradientStorage(embeddings)

                num_adv_passage_tokens = target_d_inputs['input_ids'].shape[1]
                adv_passage_ids = [ctx_tokenizer.mask_token_id] * (num_adv_passage_tokens)
                adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

                best_adv_passage_ids = adv_passage_ids.clone()
                best_sim = evaluate_acc_batch(ctx_encoder, ctx_encoder, get_emb, adv_passage_ids, adv_passage_attention,
                                              adv_passage_token_type, target_d_inputs, score_function)
                target_p_emb = get_emb(ctx_encoder, target_d_inputs, score_function=score_function).detach().cpu().numpy()
                target_p_emb = torch.from_numpy(target_p_emb).float().to('cuda')
                adv_list = []
                for epoch in range(attack_args.num_attack_epochs):
                    print('epoch: ',epoch)
                    ctx_encoder.zero_grad()
                    p_sent = {'input_ids': adv_passage_ids,
                              'attention_mask': adv_passage_attention,
                              # 'token_type_ids': adv_passage_token_type
                              }
                    p_emb = get_emb(ctx_encoder, p_sent,score_function= score_function)
                    # Compute loss
                    sim = torch.mm(target_p_emb, p_emb.T)  # [b x k]
                    loss = sim.mean()
                    # print('loss', loss.cpu().item())
                    loss.backward()
                    temp_grad = embedding_gradient.get()
                    grad = temp_grad.sum(dim=0)
                    token_to_flip, candidates = hotflip_candidate( grad, embeddings)
                    current_score, candidate_scores = hotflip_candidate_score( candidates, target_p_emb, None, get_emb, ctx_encoder, ctx_encoder,
                        adv_passage_ids, adv_passage_attention, adv_passage_token_type, token_to_flip, score_function)
                    print('current_score', current_score, "candidate_scores", candidate_scores,)
                    # if find a better one, update
                    if (candidate_scores > current_score).any() :
                        # logger.info('Better adv_passage detected.')
                        best_candidate_score = candidate_scores.max()
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                        # print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
                        text = ctx_tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True)
                        # print('Current adv_text: ', text)
                        best_adv_passage_ids = adv_passage_ids.clone()
                        adv_text = ctx_tokenizer.decode(best_adv_passage_ids[0], skip_special_tokens=True)

                        adv_text_inputs = ctx_tokenizer(adv_text, return_tensors='pt', padding=True, truncation=True,encode_passage=True,
                                                    max_length=data_args.passage_max_len)

                        d_adv_score = embedding_score_two(recon_model, target_d_inputs, adv_text_inputs,
                                                          score_function=score_function)
                        adv_list.append({
                            'epoch': epoch,
                            'd_adv_score': d_adv_score,
                            'adv_text': adv_text,
                        })
                # Sort the list by 'd_adv_score' in descending order
                adv_list = sorted(adv_list, key=lambda x: x['d_adv_score'], reverse=True)
                adv_text = adv_list[0]['adv_text']
                print('adv_text: ', adv_text)

            elif attack_args.method == 'random_token':
                # random token replacement attack
                adv_text_inputs = copy.deepcopy(target_d_inputs)

                tokenizer_vocab_size = ctx_tokenizer.vocab_size
                replacement_rate = attack_args.random_token_replacement_rate
                input_ids = target_d_inputs['input_ids']

                batch_size, seq_length = input_ids.shape
                num_tokens_to_replace = int(replacement_rate * seq_length)

                replacement_mask = torch.rand(input_ids.shape, device=input_ids.device) < replacement_rate

                random_tokens = torch.randint(
                    low=0,
                    high=tokenizer_vocab_size,
                    size=input_ids.shape,
                    device=input_ids.device
                )

                adv_new_input_ids = torch.where(replacement_mask, random_tokens, input_ids)

                adv_text_inputs['input_ids'] = adv_new_input_ids
                adv_text_list = ctx_tokenizer.batch_decode(adv_text_inputs['input_ids'], skip_special_tokens=True)
                adv_text = adv_text_list[0]
                print('adv_text: ', adv_text)

            elif attack_args.method == 'random_noise':

                mean = 0
                std = 1
                beta = attack_args.random_noise_rate
                with torch.no_grad():
                    target_d_emb = recon_model.encode_passage(target_d_inputs)
                    gaussian_noise = torch.normal(mean, std, size=target_d_emb.shape)
                    encode_embedding_noise = target_d_emb + gaussian_noise.to(device) * beta


                    encode_embedding_noise = recon_model.transfer_linear(encode_embedding_noise)
                    decoder_outputs = recon_model.context_decoder(
                        inputs_embeds=encode_embedding_noise,
                        labels=target_d_inputs['input_ids'],
                    )
                    cross_entropy_loss_adv = decoder_outputs['loss']
                    decoder_outputs_log = decoder_outputs['logits']

                    adv_text_index = torch.argmax(decoder_outputs_log, dim=-1)
                    adv_text_list = ctx_tokenizer.batch_decode(adv_text_index, skip_special_tokens=False)
                    adv_text = adv_text_list[0]
                    print('adv_text: ', adv_text)
            else:
                raise NotImplementedError


            attack_result_dic[index]['adv_text'] = adv_text

        save_path = "output/results_corpus_attack/%s-generate/%s/%s/attack_rate_%s-run_id_%d-seed%d.json" % (
         attack_args.method, attack_args.attack_dataset, attack_args.attack_mode_code, attack_args.attack_rate, attack_args.run_id, training_args.seed)
        path_exist(save_path)
        with open(save_path, 'w') as f:
            json.dump(attack_result_dic, f, indent=4)

if __name__ == "__main__":
    main()