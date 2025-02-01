import os
from dataclasses import dataclass, field
from typing import Optional, List,Literal
from typing import Any, Dict, List, Optional, Union
from utils.utils import path_change_between_windows_linux,path_join
import torch
from transformers import Trainer, TrainingArguments

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

# class TrainArguments(TrainingArguments):
@dataclass
class TrainingArguments(TrainingArguments):

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": " "})
    seed: int = field(default=2025,metadata={"help": "random seed， 2024, 2025, 2026"})
    num_train_epochs : Optional[int] = field(default=5, metadata={"help": " "})
    learning_rate: float = field(default=5e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW if we apply some."})
    do_train: bool = field(default = True, metadata={"help": " "})
    do_eval: Optional[bool] = field(default=False)
    output_dir: str = field(default=None, metadata={"help": "Path to output"})
    save_steps: int = field(default=50000)
    logging_steps: float = field(
        default=100,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    report_to: Union[None, str, List[str]] = field(
        default="wandb", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    save_total_limit: Optional[int] = field( default=10,  metadata={ "help": "Limit the total amount of checkpoints. "
        "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints." }    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

@dataclass
class ModelArguments:
    recon_model_code:str = field(default='simlm-msmarco', metadata={"help": "contriever, contriever-msmarco, simlm-msmarco, dpr-single, dpr-multi, ance, tas-b, dragon, retromae_msmarco, retromae_msmarco_finetune, retromae_msmarco_distill " })
    decoder_base_model: str = field(default="google-bert/bert-large-uncased",metadata={"help": "google-bert/bert-base-uncased , google-bert/bert-large-uncased"})


@dataclass
class DataArguments:
    dataset_name: str = field(
        default = 'msmarco', metadata={"help": "BEIR dataset name"}
    )
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    query_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for query"}
    )
    passage_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for passage"}
    )
    append_eos_token: bool = field(
        default=False, metadata={"help": "append eos token to query and passage, this is currently used for repllama"}
    )

@dataclass
class AttackArguments:
    method: str = field(default="unsupervised", metadata={"help": "hotflip or  unsupervised random_noise random_token pgd_attack"})
    attack_dataset: str = field(default="nfcorpus", metadata={"help": "BEIR dataset to evaluate,  nq, nq-train, msmarco  arguana fiqa trec-covid  nfcorpus trec_dl20 trec_dl19"})
    attack_split: str = field(default='test', metadata={"help": "train, test"})
    attack_mode_code: str = field(default='simlm-msmarco', metadata={"help": "contriever, contriever-msmarco, dpr-single, dpr-multi,ance tas-b simlm-msmarco "})
    attack_number: int = field(default=100, metadata={"help": "the number of attack"})
    num_attack_epochs: Optional[int] = field(default=3000, metadata={"help": " "})
    lm_loss_clip_max: Optional[float] = field(default=5.0, metadata={"help": " "})
    perplexity_model_name_or_path: Optional[str] = field(default='meta-llama/Llama-3.2-1B', metadata={"help": "'meta-llama/Llama-3.2-1B' 'gpt2' "})

    random_token_replacement_rate: float = field(default=0.3, metadata={"help": "random token attack"})
    random_noise_rate: float = field(default=0.5
                                     , metadata={"help": "random noise attack"})
    attack_rate: float = field(default=0.0005,metadata={"help":" corpus posising rate"})
    run_id : int = field(default=0, metadata={"help": "Divided into 20 gpus, so the value range is 0-19"})