'''
Recover text from matrix vector representation
'''
from transformers import set_seed,BertForTokenClassification
# Load the dataset
import wandb
import os
import sys
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from utils.utils import model_code_to_cmodel_name
from utils.load_model import prefix_tokenizer
from utils.load_data import load_beir_data
from utils.data_loader import BeirEncodeDataset,TrainCollator
from models.model_reconstruction import BaseRecontructionModel, ReconstructionTrainer
from arguments_new import ModelArguments, DataArguments, TrainingArguments
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="UCP_recon_model_training",
        # track hyperparameters and run metadata
        # config=vars(args),
    )

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
    training_args.output_dir =  f"output/recon_models/{model_args.recon_model_code}/decoder_{(model_args.decoder_base_model).split('/')[-1]}"

    # Load ctx encoder model
    recon_model_c_path = model_code_to_cmodel_name.get(model_args.recon_model_code, model_args.recon_model_code)
    ctx_encoder = AutoModel.from_pretrained(recon_model_c_path)
    tokenizer = AutoTokenizer.from_pretrained(recon_model_c_path) # By default, all tokenizers are bert-based.
    tokenizer = prefix_tokenizer(model_args.recon_model_code, tokenizer=tokenizer)

    ctx_encoder.eval()

    corpus, query, qrels = load_beir_data(data_args.dataset_name, split='test')  # 'test' We use corpus for training
    corpus_dataset = BeirEncodeDataset(corpus=corpus, tokenizer=tokenizer, max_length=data_args.passage_max_len)
    data_collator = TrainCollator(encode_query=False, encode_passage=True, max_length=data_args.passage_max_len, tokenizer = tokenizer)

    print("corpus_dataloader")
    # Load the reconstruction model
    ctx_decoder = BertForTokenClassification.from_pretrained(model_args.decoder_base_model,num_labels=tokenizer.vocab_size)
    recon_model = BaseRecontructionModel(ctx_encoder, ctx_decoder)

    trainer = ReconstructionTrainer(
        model=recon_model,
        args=training_args,
        train_dataset=corpus_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    wandb.finish()

if __name__ == '__main__':
    main()