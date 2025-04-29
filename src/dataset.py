import torch
from mpmath import convert
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional, Tuple


class CoralItemDataset(Dataset):
    def __init__(
            self,
            item_text: dict[str, List[str]],
            base_tokenizer_name: Optional[str] = 'nvidia/NV-Embed-v1',
            max_length: Optional[int] = 512,
            truncation_side: Optional[str] = 'right',
    ):
        super().__init__()
        self.item_texts = item_text['item'] if 'item' in item_text else None

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, truncation_side=truncation_side)

    def __len__(self):
        return len(self.item_texts)

    def __getitem__(self, idx):
        tokenized_i = self.tokenizer(
            self.item_texts[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )

        new_tokenized_i = dict()
        if tokenized_i is not None:
            for key, value in tokenized_i.items():
                new_tokenized_i[f'item_{key}'] = value

        new_tokenized_i = {key: val.squeeze(0) for key, val in new_tokenized_i.items() if val is not None}
        return new_tokenized_i

    def collate_fn(self, batch):
        item_input_ids, item_attention_mask = [], []

        for item in batch:
            item_input_ids.append(item['item_input_ids'])
            item_attention_mask.append(item['item_attention_mask'])

        # pad into max length
        item_input_ids = torch.nn.utils.rnn.pad_sequence(item_input_ids, batch_first=True,
                                                         padding_value=self.tokenizer.pad_token_id)
        item_attention_mask = torch.nn.utils.rnn.pad_sequence(item_attention_mask, batch_first=True,
                                                              padding_value=0)

        return {
            'item_input_ids': item_input_ids if self.item_texts else None,
            'item_attention_mask': item_attention_mask if self.item_texts else None,
        }


class CoralDataset(Dataset):
    def __init__(
            self,
            user_text: dict[str, List[str]],
            gt_ids: List[int]|List[List[int]],
            base_tokenizer_name: Optional[str] = 'nvidia/NV-Embed-v1',
            max_length: Optional[int] = 512,
            truncation_side: Optional[str] = 'right',
    ):
        super().__init__()
        self.conv_texts = user_text['c'] if 'c' in user_text else None
        self.like_texts = user_text['l'] if 'l' in user_text else None
        self.dislike_texts = user_text['d'] if 'd' in user_text else None

        self.gt_ids = gt_ids
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, truncation_side=truncation_side)

        self.len = len(self.conv_texts)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tokenized_conv = self.tokenizer(
            self.conv_texts[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ) if self.conv_texts is not None else None

        tokenized_like = self.tokenizer(
            self.like_texts[idx],
            max_length=self.max_length // 2,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ) if self.like_texts is not None else None

        tokenized_dislike = self.tokenizer(
            self.dislike_texts[idx],
            max_length=self.max_length // 2,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ) if self.dislike_texts is not None else None

        tokenized_u = dict()
        if tokenized_conv is not None:
            for key, value in tokenized_conv.items():
                tokenized_u[f'conv_{key}'] = value

        if tokenized_like is not None:
            for key, value in tokenized_like.items():
                tokenized_u[f'like_{key}'] = value

        if tokenized_dislike is not None:
            for key, value in tokenized_dislike.items():
                tokenized_u[f'dislike_{key}'] = value

        tokenized_u = {key: val.squeeze(0) for key, val in tokenized_u.items() if val is not None}
        gt_id = self.gt_ids[idx]

        return idx, gt_id, tokenized_u

    def collate_fn(self, batch):
        batch_user_id = [idx for idx, _, _ in batch]
        batch_gt_id = [idx for _, idx, _ in batch]
        if self.conv_texts:
            conv_input_ids, conv_attention_mask = [], []
        if self.like_texts:
            like_input_ids, like_attention_mask = [], []
        if self.dislike_texts:
            dislike_input_ids, dislike_attention_mask = [], []

        for _, _, tokenized_user in batch:
            if self.conv_texts:
                conv_input_ids.append(tokenized_user['conv_input_ids'])
                conv_attention_mask.append(tokenized_user['conv_attention_mask'])
            if self.like_texts:
                like_input_ids.append(tokenized_user['like_input_ids'])
                like_attention_mask.append(tokenized_user['like_attention_mask'])
            if self.dislike_texts:
                dislike_input_ids.append(tokenized_user['dislike_input_ids'])
                dislike_attention_mask.append(tokenized_user['dislike_attention_mask'])

        # pad into max length
        if self.conv_texts:
            conv_input_ids = torch.nn.utils.rnn.pad_sequence(conv_input_ids, batch_first=True,
                                                             padding_value=self.tokenizer.pad_token_id)
            conv_attention_mask = torch.nn.utils.rnn.pad_sequence(conv_attention_mask, batch_first=True,
                                                                  padding_value=0)
        if self.like_texts:
            like_input_ids = torch.nn.utils.rnn.pad_sequence(like_input_ids, batch_first=True,
                                                             padding_value=self.tokenizer.pad_token_id)
            like_attention_mask = torch.nn.utils.rnn.pad_sequence(like_attention_mask, batch_first=True,
                                                                  padding_value=0)
        if self.dislike_texts:
            dislike_input_ids = torch.nn.utils.rnn.pad_sequence(dislike_input_ids, batch_first=True,
                                                                padding_value=self.tokenizer.pad_token_id)
            dislike_attention_mask = torch.nn.utils.rnn.pad_sequence(dislike_attention_mask, batch_first=True,
                                                                     padding_value=0)
        return batch_user_id, batch_gt_id, {
            'conv_input_ids': conv_input_ids if self.conv_texts else None,
            'conv_attention_mask': conv_attention_mask if self.conv_texts else None,
            'like_input_ids': like_input_ids if self.like_texts else None,
            'like_attention_mask': like_attention_mask if self.like_texts else None,
            'dislike_input_ids': dislike_input_ids if self.dislike_texts else None,
            'dislike_attention_mask': dislike_attention_mask if self.dislike_texts else None
        }
