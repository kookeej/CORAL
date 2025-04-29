import os
import copy
import numpy as np
from mpmath import convert
from peft import LoraConfig, LoraModel, get_peft_model, PeftModel

from tqdm import tqdm
from typing import Optional, List, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel
)


def _mean_pooling(output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(output).float()  # same with sentence_embeddings
    sum_output = torch.sum(output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_output = sum_output / sum_mask

    return mean_output


class CoralEncoder(nn.Module):
    def __init__(self, base_model_name: str, alpha:float, beta:float):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.alpha, self.beta = alpha, beta

        self.configure_lora()

    def configure_lora(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            target_modules=target_modules,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()


    def forward(
            self,
            item_input_ids: Optional[torch.Tensor] = None,
            item_attention_mask: Optional[torch.Tensor] = None,
            conv_input_ids: Optional[torch.Tensor] = None,
            conv_attention_mask: Optional[torch.Tensor] = None,
            like_input_ids: Optional[torch.Tensor] = None,
            like_attention_mask: Optional[torch.Tensor] = None,
            dislike_input_ids: Optional[torch.Tensor] = None,
            dislike_attention_mask: Optional[torch.Tensor] = None,
    ):

        item_output = self.model(
            input_ids=item_input_ids,
            attention_mask=item_attention_mask,
        ) if item_input_ids is not None else None

        conv_output = self.model(
            input_ids=conv_input_ids,
            attention_mask=conv_attention_mask,
        ) if conv_input_ids is not None else None

        like_output = self.model(
            input_ids=like_input_ids,
            attention_mask=like_attention_mask,
        ) if like_input_ids is not None else None

        dislike_output = self.model(
            input_ids=dislike_input_ids,
            attention_mask=dislike_attention_mask,
        ) if dislike_input_ids is not None else None

        item_embedding = _mean_pooling(item_output['sentence_embeddings'],
                                             attention_mask=item_attention_mask) if item_output is not None else None
        conv_embedding = _mean_pooling(conv_output['sentence_embeddings'],
                                        attention_mask=conv_attention_mask) if conv_output is not None else None
        like_embedding = _mean_pooling(like_output['sentence_embeddings'],
                                             attention_mask=like_attention_mask) if like_output is not None else None
        dislike_embedding = _mean_pooling(dislike_output['sentence_embeddings'],
                                                attention_mask=dislike_attention_mask) if dislike_output is not None else None

        item_embedding = F.normalize(item_embedding, p=2, dim=-1) if item_embedding is not None else None
        conv_embedding = F.normalize(conv_embedding, p=2, dim=-1) if conv_embedding is not None else None
        like_embedding = F.normalize(like_embedding, p=2, dim=-1) if like_embedding is not None else None
        dislike_embedding = F.normalize(dislike_embedding, p=2,
                                         dim=-1) if dislike_embedding is not None else None

        return {
            'item_embedding': item_embedding,
            'conv_embedding': conv_embedding,
            'like_embedding': like_embedding,
            'dislike_embedding': dislike_embedding,
        }

    def save_checkpoint(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        self.model.save_pretrained(Path(ckpt_save_path))


    def load_best_checkpoint(self, ckpt_save_path):
        self.model.load_adapter(Path(ckpt_save_path), adapter_name='lora')

