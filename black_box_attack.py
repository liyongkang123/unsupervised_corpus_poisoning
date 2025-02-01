'''
Studying the effects of transfer attacks
'''

import  json
import wandb
from transformers.modeling_outputs import SequenceClassifierOutput
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from utils.load_data import load_beir_data
from utils.utils import move_to_cuda,format_passage
import config as input_config
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer,DPRQuestionEncoder,DPRQuestionEncoderTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def dragon_get_score(q_model,c_model, tokenizer, query, passage, title: str = ''  ):
    passage = format_passage(text=passage, title=title)
    q_inputs = tokenizer(query,max_length=32, padding=True, truncation=True,  return_tensors='pt')
    p_inputs = tokenizer(passage, max_length=128, padding=True, truncation=True, return_tensors='pt')
    q_inputs = move_to_cuda(q_inputs)
    p_inputs = move_to_cuda(p_inputs)
    with torch.no_grad():
        q_embeddings = q_model(**q_inputs).last_hidden_state[:, 0, :]
        p_embeddings = c_model(**p_inputs).last_hidden_state[:, 0, :]
    score = (q_embeddings * p_embeddings).sum(dim=1).item()
    return score

def simlm_reranker_score(q_model,c_model, tokenizer, query, passage , title: str ='-'):
    input = move_to_cuda(tokenizer(query, '{}: {}'.format(title, passage),   max_length=192,   padding=True, truncation=True,  return_tensors='pt'))
    outputs: SequenceClassifierOutput = q_model(**input, return_dict=True)
    score = outputs.logits[0].item()
    return score

def colbert_get_score(q_model, c_model, tokenizer, query, passage, title: str = '-' ):
    Q = q_model.queryFromText([query])
    D = q_model.docFromText([passage], bsize=32)[0]
    D_mask = torch.ones(D.shape[:2], dtype=torch.long)
    Q = move_to_cuda(Q)
    D = move_to_cuda(D)
    D_mask = move_to_cuda(D_mask)
    scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
    return scores[0]

def rankllama_get_score(q_model, c_model, tokenizer, query, passage, title: str = '-' ):
    # Tokenize the query-passage pair
    title = ""
    inputs = move_to_cuda(tokenizer(f'query: {query}', f'document: {title} {passage}', return_tensors='pt'))
    # Run the model forward
    with torch.no_grad():
        outputs = q_model(**inputs)
        logits = outputs.logits
        score = logits[0][0]
        # print(score)
    return score.item()

def dpr_get_score(q_model, c_model, tokenizer, query, passage, title: str = '' ):
    passage = format_passage(text=passage, title=title)
    q_inputs = tokenizer(query,max_length=32, padding=True, truncation=True,  return_tensors='pt')
    p_inputs = tokenizer(passage, max_length=128, padding=True, truncation=True, return_tensors='pt')
    q_inputs = move_to_cuda(q_inputs)
    p_inputs = move_to_cuda(p_inputs)
    with torch.no_grad():
        q_embeddings = q_model(**q_inputs).pooler_output
        p_embeddings = c_model(**p_inputs).pooler_output
    score = (q_embeddings * p_embeddings).sum(dim=1).item()
    return score

def main():
    args =  input_config.parse()
    print(args)
    datasets_name_list = [args.attack_dataset] #['trec_dl19', "trec_dl20",   "nq",  "quora",  "fiqa", "webis-touche2020"]
    methods_list = [args.method]  # ['random_noise', 'random_token','hotflip', 'unsupervised', ]
    # datasets_name_list = ['trec_dl19', "trec_dl20",   "nq",  "quora",  "fiqa", "webis-touche2020"]
    # methods_list = ['random_noise', 'random_token','hotflip', 'unsupervised', ]

    seed_list = [2024, 2025, 2026]

    wandb.init(
        # set the wandb project where this run will be logged
        project="UCP_attack_black_box",
        # track hyperparameters and run metadata
        config={
            'target_attack_datasets': args.attack_dataset,
            'methods': args.method,
        }
    )

    attack_mode_code = 'simlm-msmarco'
    target_model_code_list = [ "dpr", 'dragon', "intfloat/simlm-msmarco-reranker", "colbert-ir/colbertv2.0",   "castorini/rankllama-v1-7b-lora-passage" ]

    attack_split = 'test'
    learning_rate = 0.0005
    lm_loss_clip_max = 5.0
    random_noise_rate = 0.5
    random_token_replacement_rate = 0.3

    for target_model_code in target_model_code_list:
        if 'simlm' in target_model_code:
            tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-msmarco-reranker')
            q_model = AutoModelForSequenceClassification.from_pretrained('intfloat/simlm-msmarco-reranker')
            c_model = q_model
            reranker_score = simlm_reranker_score

        elif 'colbert' in target_model_code:
            checkpoint = 'colbert-ir/colbertv2.0'
            tokenizer = None
            config = ColBERTConfig.load_from_checkpoint(checkpoint)
            config.checkpoint = "/cache/models--colbert-ir--colbertv2.0/snapshots/c1e84128e85ef755c096a95bdb06b47793b13acf/"
            q_model = Checkpoint(checkpoint, colbert_config=config)
            c_model = q_model
            reranker_score = colbert_get_score

        elif 'rankllama' in target_model_code:
            # Load the tokenizer and model
            peft_model_name = 'castorini/rankllama-v1-7b-lora-passage'
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
            config = PeftConfig.from_pretrained(peft_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                            num_labels=1)
            q_model = PeftModel.from_pretrained(base_model, peft_model_name)
            q_model = q_model.merge_and_unload()
            c_model = q_model
            reranker_score = rankllama_get_score

        elif 'dragon' in target_model_code:
            q_model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
            c_model = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
            tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
            reranker_score = dragon_get_score

        elif 'dpr' in target_model_code:
            q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            c_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            reranker_score = dpr_get_score

        else:
            raise NotImplementedError

        q_model.to(device)
        c_model.to(device)
        q_model.eval()
        c_model.eval()

        for eval_datset in datasets_name_list:

            corpus, query, qrels = load_beir_data(eval_datset, split='test')
            for method in methods_list:
                ASR=[]
                for seed in seed_list:
                    print(eval_datset, method, seed)
                    if method == 'random_noise':
                        attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                               f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}"
                                               f"_noise_rate-{random_noise_rate}.json")
                    elif method == 'random_token':
                        attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                               f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}"
                                               f"random_token-{random_token_replacement_rate}.json")

                    elif method == 'hotflip':
                        attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                               f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}.json")

                    elif method == 'unsupervised':
                        attack_results_path = (f"output/attack_results/{method}/{eval_datset}/"
                                               f"{attack_mode_code}/{attack_split}_top1_attack_lr-{learning_rate}_lm_loss_clip_max-{lm_loss_clip_max}_seed-{seed}.json")
                    print(attack_results_path)


                    with open(attack_results_path, 'r', encoding='utf-8') as file:
                        attack_results = json.load(file)


                    for q in attack_results.keys():
                        top_1_target_text = attack_results[q]['top_1_target_text']
                        adv_text = attack_results[q]['adv_text']
                        q_query = attack_results[q]['query_text']

                        qrels_q = qrels[q]
                        qrels_q_score = []
                        for r_d_in_q in qrels_q:
                            if qrels_q[r_d_in_q] > 0: ## Indicates that this document is relevant
                                title_passage_dic =corpus.get(r_d_in_q, {"title": "", "text": ""})
                                title = title_passage_dic.get("title", "")
                                passage = title_passage_dic.get("text", "")

                                r_d_in_q_score = reranker_score(q_model, c_model, tokenizer, q_query,  passage,title)
                                qrels_q_score.append(r_d_in_q_score)

                        adv_text_score = reranker_score(q_model, c_model, tokenizer, q_query, adv_text)
                        print("q: ", q," adv_text_score: ",adv_text_score)

                        if adv_text_score > min(qrels_q_score):
                            ASR.append(1)
                        else:
                            ASR.append(0)

                ASR_percent = sum(ASR) / len(ASR) # # This value has been averaged across different seeds
                print(f"{eval_datset} {method}, attack reranker{target_model_code} ASR_percent: {ASR_percent}")

if __name__ == '__main__':
    main()