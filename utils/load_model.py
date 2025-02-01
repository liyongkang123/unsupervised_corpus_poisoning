import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer,AutoModel , BertTokenizerFast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import torch
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor
import torch.nn.functional as F
import transformers
from utils.utils import model_code_to_qmodel_name,model_code_to_cmodel_name,prefix_to_model_code,pooling_to_model_code

from peft import PeftModel, PeftConfig

def get_llama_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    return model

def l2_normalize(x: torch.Tensor): # for simlm  model
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor: # for E5 model
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":  # average pooling
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class prefix_tokenizer:
    def __init__(self, model_code, tokenizer=None):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code], use_fast=True)
        self.prefix_dic = prefix_to_model_code.get(model_code, None)

    def _add_prefix(self, input_texts, prefix):
        #Internal method: Add prefix to input text """
        if isinstance(input_texts, str):
            return prefix + input_texts
        elif isinstance(input_texts, list):
            return [prefix + text for text in input_texts]
        else:
            raise TypeError("Input texts must be either a string or a list of strings.")

    def __call__(self, input_texts, encode_query=False, encode_passage=False, **kwargs):
        # Designed for single input (such as a string or list),
        if self.prefix_dic:
            if not encode_query and not encode_passage:
                raise ValueError("Either encode_query or encode_passage must be True if prefix is provided.")

            if encode_query:
                prefix = self.prefix_dic.get('query_prefix', '')
            elif encode_passage:
                prefix = self.prefix_dic.get('document_prefix', '')
            else:
                prefix = ''
        else:
            prefix = ''

        input_texts = self._add_prefix(input_texts, prefix) if prefix else input_texts
        result = self.tokenizer(input_texts, **kwargs)
        result.pop('token_type_ids', None) # # Remove token_type_ids, in favor of DistilBert model, since it does not require token_type_ids. This can only be done when we use single sentence tasks.
        return result

    def __getattr__(self, attr):
        tokenizer = super().__getattribute__('tokenizer')
        if hasattr(tokenizer, attr):
            return getattr(tokenizer, attr)
        raise AttributeError(f"'PrefixTokenizer' object has no attribute '{attr}'")