import sys, os
import os

from utils.utils import move_to_cuda

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, set_seed,BertForTokenClassification,set_seed,HfArgumentParser
import torch.nn.functional as F
import numpy as np
import torch
from .model_reconstruction import BaseRecontructionModel
import logging
logger = logging.getLogger(__name__)

class NoiseModel(nn.Module):
    def __init__(self, in_dim: object, out_dim: object, hidden_dim: object, num_layers: object) -> object:
        """
        Initialize the noise model.
        Parameters:
        in_dim (int): input dimension
        out_dim (int): output dimension
        hidden_dim (int): hidden layer dimension
        num_layers (int): number of hidden layers
        """
        super(NoiseModel, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class UnsupervisedAttack(nn.Module):
    '''
    frozen encoder model and decoder model
    '''

    def __init__(self, model:BaseRecontructionModel, tokenizer, seed=2025):
        super(UnsupervisedAttack, self).__init__()
        set_seed(seed)
        self.recon_model = model
        self.recon_model.eval()
        self.tokenizer = tokenizer

        self.noise_model = NoiseModel(in_dim=768, out_dim=768, hidden_dim=768, num_layers=2)
        self.noise_model.train()

        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        # Get the embedding matrix of BERT
        self.encoder_embedding_matrix = self.recon_model.context_encoder.embeddings.word_embeddings.weight  # (vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(self.recon_model.context_encoder.config.hidden_size,  )

        self._freeze_context_encoder()

    def _freeze_context_encoder(self):
        for param in self.recon_model.context_encoder.parameters():
            param.requires_grad = False
        for param in self.recon_model.context_decoder.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        print('The model is initializing parameters')
        for layer in self.noise_model.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs, adv_emebdding=None):
        encode_embedding = self.recon_model.encode_passage(inputs)
        encode_embedding_noise = self.noise_model(encode_embedding)
        mse_loss_noise = self.mse_loss(encode_embedding, encode_embedding_noise)

        # decoder
        encode_embedding_noise = self.recon_model.transfer_linear(encode_embedding_noise)
        decoder_outputs = self.recon_model.context_decoder(
            inputs_embeds=encode_embedding_noise,
            labels=inputs['input_ids'],
        )
        cross_entropy_loss_adv = decoder_outputs['loss']
        decoder_outputs_log = decoder_outputs['logits']

        adv_text_index = torch.argmax(decoder_outputs_log, dim=-1)
        adv_text_list = self.tokenizer.batch_decode(adv_text_index, skip_special_tokens=False)

        return {
            "mse_loss_noise": mse_loss_noise,
            'adv_text': adv_text_list[0],
            "cross_entropy_loss_adv": cross_entropy_loss_adv,
        }