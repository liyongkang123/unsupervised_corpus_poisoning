# Define the reconstruction model for the reconstruction of input text
import sys, os
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from transformers import BartModel, AutoTokenizer, BartPretrainedModel, BartConfig, BertConfig, AutoConfig, \
    BartForCausalLM, AutoModel, BartForConditionalGeneration, T5ForConditionalGeneration, BertForTokenClassification, \
    BertModel
from transformers import PreTrainedModel, AutoModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import evaluate
from torch import nn, Tensor
from sklearn.metrics import f1_score
import torch.distributed as dist
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Dict, Optional
from torch import Tensor
import logging
logger = logging.getLogger(__name__)

from transformers.trainer import Trainer, TRAINING_ARGS_NAME

class BaseRecontructionModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 context_encoder: PreTrainedModel,
                 decoder: PreTrainedModel = None,
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super(BaseRecontructionModel, self).__init__()
        self.context_encoder = context_encoder

        self.normalize = normalize
        self.temperature = temperature
        self.pooling = 'all'
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.context_decoder = decoder
        self.transfer_linear = nn.Sequential(
            nn.Linear(self.context_encoder.config.hidden_size, self.context_decoder.config.hidden_size, bias=True), # for bert large, 只用一层linear
        )
        self.transfer_linear.train()
        self.context_decoder.train()
        self.loss_fct = CrossEntropyLoss()

        self.context_encoder.eval()
        self.freeze_context_encoder()

        if 'dpr' in  self.context_encoder.config.architectures[0].lower():
            self.dpr = True
        else:
            self.dpr = False

    def freeze_context_encoder(self):
        for param in self.context_encoder.parameters():
            param.requires_grad = False

    def forward(self, inputs: Dict[str, Tensor] = None):

        with torch.no_grad():
            encode_embedding=self.encode_passage(inputs)
        # decoder
        encode_embedding = self.transfer_linear(encode_embedding)
        outputs = self.context_decoder(inputs_embeds=encode_embedding, labels=inputs['input_ids'])
        loss = outputs['loss']
        logits = outputs['logits']
        return logits,loss

    def encode_query(self, qry):
        if self.dpr:
            query_hidden_states = self.context_encoder(**qry, output_hidden_states=True, return_dict=True)
            query_hidden_states = query_hidden_states.hidden_states[-1]
        else:
            query_hidden_states = self.context_encoder(**qry, return_dict=True)
            query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)

    def _pooling(self, last_hidden_state, attention_mask):
        assert self.pooling == "all" #By default, all hidden states are used here

        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif self.pooling in ['all']:
            reps = last_hidden_state
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')

        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


class ReconstructionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ReconstructionTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits, loss = model(inputs)
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save model
        if hasattr(self.model, "module"):  # save the model w/o the DDP wrapper
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        supported_classes = (BaseRecontructionModel,)
        if not isinstance(model_to_save, supported_classes):
            raise ValueError(f"Unsupported model class {model_to_save}")
        else:
            model_to_save.context_decoder.save_pretrained(output_dir, safe_serialization=True)
            torch.save(model_to_save.transfer_linear.state_dict(), os.path.join(output_dir, "transfer_linear.pt"))

        torch.save(torch.get_rng_state(), os.path.join(output_dir, "rng_state.pth"))
        torch.save(torch.cuda.get_rng_state_all(), os.path.join(output_dir, "cuda_rng_state.pth"))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))