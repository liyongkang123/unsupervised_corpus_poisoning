'''
Evaluation of the reconstruction model
'''

import os
import torch
import sys
from arguments_new import ModelArguments, DataArguments, TrainingArguments
from transformers import (
    HfArgumentParser,
)
import wandb
from transformers import AutoTokenizer, AutoModel, set_seed,BertForTokenClassification
from utils.load_data import load_beir_data
from utils.data_loader import BeirEncodeDataset
from models.model_reconstruction import BaseRecontructionModel
from utils.utils import model_code_to_cmodel_name, move_to_cuda
from utils.load_model import prefix_tokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from torch.utils.data import DataLoader

def Recontruction_evaluate_test(model, dataloader, device):
    '''
    Evaluate the model on the given dataloader
    '''
    model.to(device)
    model.eval()
    all_y_true = []
    all_y_pred = []

    accuracy_results, precision_results, recall_results,f1_results = [],[],[],[]
    for batch_idx_test, batch_data_test in enumerate(tqdm(dataloader, desc="Test", leave=False)):
        with torch.no_grad():
            batch_data_test = move_to_cuda(batch_data_test)
            logits,_ = model(batch_data_test)
            predicted_token_list = logits.argmax(-1)
            # Assume that y_true and y_pred are the true label and predicted label respectively.
            y_true = batch_data_test['input_ids'].reshape(-1).cpu().numpy().tolist()
            y_pred = predicted_token_list.reshape(-1).cpu().numpy().tolist()
            # Append to overall lists
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            assert len(y_true) == len(y_pred)
    precision = precision_score(all_y_true, all_y_pred, average='macro')
    recall = recall_score(all_y_true, all_y_pred, average='macro')
    accuracy = accuracy_score(all_y_true, all_y_pred)
    f1_scores = f1_score(all_y_true, all_y_pred, average='macro')

    return precision, recall, accuracy, f1_scores

def main():

    wandb.init(
        # set the wandb project where this run will be logged
        project="UCP_recon_model_eval",
        # track hyperparameters and run metadata
        # config=vars(args),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments


    wandb.config.update(vars(model_args))
    wandb.config.update(vars(data_args))
    wandb.config.update(vars(training_args))

    set_seed(training_args.seed)  # set seed for reproducibility

    # Load the model
    recon_model_c_path = model_code_to_cmodel_name.get(model_args.recon_model_code, model_args.recon_model_code)
    ctx_encoder = AutoModel.from_pretrained(recon_model_c_path)
    tokenizer = AutoTokenizer.from_pretrained(recon_model_c_path)
    tokenizer = prefix_tokenizer(model_args.recon_model_code, tokenizer=tokenizer)

    ctx_encoder.eval()

    load_model_path = f"output/recon_models/{model_args.recon_model_code}/decoder_{(model_args.decoder_base_model).split('/')[-1]}/checkpoint-86350"
    decoder = BertForTokenClassification.from_pretrained(load_model_path,num_labels=tokenizer.vocab_size)
    decoder.eval()

    model = BaseRecontructionModel(context_encoder=ctx_encoder, decoder=decoder)
    model.transfer_linear.load_state_dict(torch.load(f"{load_model_path}/transfer_linear.pt"))

    eval_datasets_list=['nq',] # test on the nq dataset
    accuracy_mean_all = []
    accuracy_variance_all = []
    f1_mean_all = []
    f1_variance_all = []

    for sub_eval_data in eval_datasets_list:
        corpus, query, qrels = load_beir_data(sub_eval_data, split='test')
        corpus_dataset = BeirEncodeDataset(corpus=corpus, tokenizer=tokenizer, max_length=data_args.passage_max_len)
        corpus_dataloader = DataLoader(corpus_dataset, batch_size=training_args.per_gpu_eval_batch_size, shuffle=True,
                                       num_workers=2, collate_fn=corpus_dataset.collate_fn)
        precision, recall, accuracy, f1_scores = Recontruction_evaluate_test(model, corpus_dataloader,device)

        print(f"Eval dataset: {sub_eval_data}. Eval model: {model_args.recon_model_code}")
        print(f"Precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1_scores: {f1_scores}")

if __name__ == '__main__':
    main()