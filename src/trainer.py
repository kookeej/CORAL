from typing import Optional, Literal, List, Dict

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

torch.set_float32_matmul_precision("medium")
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from torch.amp import autocast

from .metric import (
    recall_at_k,
    ndcg_at_k
)

class CoralTrainer:
    def __init__(
            self,
            model,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            test_dataloader: DataLoader,
            train_infer_dataloader: DataLoader,
            item_dataloader: DataLoader,
            item_num: int,
            train_conv_gt_ids: List[List],
            train_gt_ids: List,
            valid_gt_ids: List,
            test_gt_ids: List,
            optimizer: Literal['Adam', 'AdamW'],
            scheduler: Literal[
                'get_linear_schedule_with_warmup', 'get_constant_schedule_with_warmup', 'get_cosine_schedule_with_warmup'],
            accumulation_steps: int,
            training_args: Dict,
            wandb_logger: wandb.sdk.wandb_run.Run | None = None
    ):

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.train_infer_dataloader = train_infer_dataloader
        self.item_dataloader = item_dataloader

        self.train_conv_gt_ids = train_conv_gt_ids
        self.train_conv_gt_mask = torch.zeros(len(train_conv_gt_ids), item_num, dtype=torch.bool)
        for idx, doc_ids in enumerate(train_conv_gt_ids):
            self.train_conv_gt_mask[idx, doc_ids] = True

        self.train_gt_ids = train_gt_ids
        self.valid_gt_ids = valid_gt_ids
        self.test_gt_ids = test_gt_ids

        self.optimizer = getattr(optim, optimizer)
        self.scheduler = getattr(transformers, scheduler)
        self.accumulation_steps = accumulation_steps

        self.training_args = training_args
        self.wandb_logger = wandb_logger

        self.loss_fn = nn.CrossEntropyLoss()

        self.item_embeddings = None
        self.negative_samples = None

    def _set_optimizer(self, learning_rate: float):
        return self.optimizer(self.model.parameters(), lr=learning_rate)

    def _set_scheduler(self, optimizer, num_warmup_steps: Optional[int] = None,
                       num_training_steps: Optional[int] = None):
        if num_warmup_steps is None:
            num_warmup_steps = len(self.train_dataloader) * 0.1
        if num_training_steps is None:
            num_training_steps = len(self.train_dataloader) * self.training_args['epochs']
        return self.scheduler(optimizer, num_training_steps, num_warmup_steps)

    @torch.no_grad()
    def _encode_items(self, item_dataloader, device):
        self.model.eval()
        self.model.to(device)
        with (autocast(dtype=torch.bfloat16, enabled=self.training_args['bf16'], device_type=device)):
            item_encoded = []
            for batch in tqdm(item_dataloader, desc="Encoding Items..."):
                inputs = {key: val.squeeze(1).to(device) if val is not None else val for key, val in batch.items()}
                outputs = self.model(**inputs)
                item_encoded.extend(outputs[f'item_embedding'].cpu())
            return torch.stack(item_encoded, dim=0)  # (item_size, hidden_size)

    @torch.no_grad()
    def _encode_conv(self, train_infer_dataloader, device):
        self.model.eval()
        self.model.to(device)
        conv_encoded = []
        with (autocast(dtype=torch.bfloat16, enabled=self.training_args['bf16'], device_type=device)):
            for batch in tqdm(train_infer_dataloader, desc="Encoding Conv..."):
                _, _, batch = batch
                inputs = {key: val.squeeze(1).to(device) if val is not None else val for key, val in batch.items()}
                outputs = self.model(**inputs)
                conv_encoded.extend(outputs["conv_embedding"].cpu())
            return torch.stack(conv_encoded, dim=0)  # (train_size, hidden_size)

    @torch.no_grad()
    def _sample_negatives(self, negative_num, device):
        conv_embedding = self._encode_conv(self.train_infer_dataloader, device=device)
        score = torch.matmul(conv_embedding, self.item_embeddings.T)

        # mask multiple positive items
        score.masked_fill_(self.train_conv_gt_mask, float('-inf'))
        score = torch.softmax(score, dim=-1)
        negative_list = torch.multinomial(score, num_samples=negative_num,
                                          replacement=False)
        return negative_list # (query_size, negative_num)

    def _ranking_loss(self, output_u, batch_item, batch_label, device):
        rep_u_conv = output_u["conv_embedding"]  # (batch_size, hidden_size)

        total_score = torch.zeros(rep_u_conv.size(0), batch_item.size(1)).to(device)
        if 'c' in self.training_args['query_used_info']:
            total_score += torch.bmm(rep_u_conv.unsqueeze(1), batch_item.permute(0, 2, 1)).squeeze(1)
        if 'l' in self.training_args['query_used_info']:
            rep_u_like = output_u["like_embedding"]  # (batch_size, hidden_size)
            total_score += self.model.alpha * torch.bmm(rep_u_like.unsqueeze(1), batch_item.permute(0, 2, 1)).squeeze(1)
        if 'd' in self.training_args['query_used_info']:
            rep_u_dislike = output_u["dislike_embedding"]
            total_score -= self.model.beta * torch.bmm(rep_u_dislike.unsqueeze(1), batch_item.permute(0, 2, 1)).squeeze(
                1)
        total_score = total_score / self.training_args['temp']
        loss = self.loss_fn(total_score, batch_label)
        return loss

    def _ranking_predict(self, output_u, device):
        rep_u_conv = output_u["conv_embedding"]  # (batch_size, hidden_size)
        similarity_scores = torch.zeros(rep_u_conv.size(0), self.item_embeddings.size(0)).to(device)
        item_embeddings = self.item_embeddings.to(device)
        if 'c' in self.training_args['query_used_info']:
            similarity_scores += (rep_u_conv @ item_embeddings.T)
        if 'l' in self.training_args['query_used_info']:
            rep_u_like = output_u["like_embedding"]
            similarity_scores += self.model.alpha * (rep_u_like @ item_embeddings.T)
        if 'd' in self.training_args['query_used_info']:
            rep_u_dislike = output_u["dislike_embedding"]
            similarity_scores -= self.model.beta * (rep_u_dislike @ item_embeddings.T)
        return similarity_scores

    def train_one_epoch(self, epoch, optimizer, scheduler, device):
        print("\n>> Epoch: ", epoch + 1)
        self.model.to(device)
        self.model.train()

        self.item_embeddings = self._encode_items(self.item_dataloader, device=device).to('cpu')
        self.negative_samples = self._sample_negatives(self.training_args['negative_sample'] - 1, device) # (query_size, negative_num)

        total_train_loss = 0
        tq = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for step, batch in enumerate(tq):
            batch_u_id, batch_gt_id, input_u = batch
            input_u = {key: val.to(device) if val is not None else val for key, val in input_u.items()}

            batch_negative_samples = self.negative_samples[batch_u_id]  # (batch_size, negative_num-1)
            batch_negative = torch.stack([self.item_embeddings[idx] for idx in batch_negative_samples], dim=0).to(
                device)  # (batch_size, negative_num-1, hidden_size)
            batch_positive = torch.stack([self.item_embeddings[idx] for idx in batch_gt_id], dim=0).to(
                device).unsqueeze(1)  # (batch_size, 1, hidden_size)
            batch_item = torch.cat([batch_positive, batch_negative],
                                   dim=1)  # (batch_size, negative_num, hidden_size)
            batch_label = torch.zeros(len(batch_u_id), dtype=torch.long).to(device)

            with autocast(dtype=torch.bfloat16, enabled=self.training_args['bf16'], device_type=device):
                output_u = self.model(**input_u)
                train_loss = self._ranking_loss(output_u, batch_item, batch_label, self.training_args, device)

            loss_item = train_loss.item()
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"train_step_loss": loss_item})

            train_loss = train_loss / self.accumulation_steps
            train_loss.backward()

            if (step + 1) % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss_item
            tq.set_postfix(loss=loss_item)

    @torch.no_grad()
    def valid_one_epoch(self, epoch, device):
        self.model.to(device)
        self.model.eval()

        total_valid_loss = 0
        total_top_k_document = []
        for step, batch in enumerate(tqdm(self.valid_dataloader, desc=f"Validation Epoch {epoch + 1}")):
            batch_u_id, batch_gt_id, input_u = batch
            input_u = {key: val.to(device) if val is not None else val for key, val in input_u.items()}

            batch_negative_samples = self.negative_samples[batch_u_id]  # (batch_size, negative_num-1)
            batch_negative = torch.stack([self.item_embeddings[idx] for idx in batch_negative_samples], dim=0).to(
                device)  # (batch_size, negative_num-1, hidden_size)
            batch_positive = torch.stack([self.item_embeddings[idx] for idx in batch_gt_id], dim=0).to(
                device).unsqueeze(1)  # (batch_size, 1, hidden_size)
            batch_item = torch.cat([batch_positive, batch_negative],
                                   dim=1)  # (batch_size, negative_num, hidden_size)
            batch_label = torch.zeros(len(batch_u_id), dtype=torch.long).to(device)

            with autocast(dtype=torch.bfloat16, enabled=self.training_args['bf16'], device_type=device):
                output_u = self.model(**input_u)
                valid_loss = self._ranking_loss(output_u, batch_item, batch_label, self.training_args, device)
                similarity_scores = self._ranking_predict(output_u, self.training_args, device)

            batch_top_k_indices = similarity_scores.topk(k=max(self.training_args['cutoff']), dim=-1).indices.tolist()

            for top_k_indices in batch_top_k_indices:
                total_top_k_document.append(top_k_indices)

            total_valid_loss += valid_loss.item()

            if self.wandb_logger is not None:
                self.wandb_logger.log({f"valid_step_loss": valid_loss.item()})
                total_valid_loss += valid_loss.item()

        print(">> Validation results:")
        performance = {}
        for k in self.training_args['cutoff']:
            recall = recall_at_k(self.valid_gt_ids, total_top_k_document, k)
            ndcg = ndcg_at_k(self.valid_gt_ids, total_top_k_document, k)
            performance[f'valid/Recall@{k}'] = recall
            performance[f'valid/NDCG@{k}'] = ndcg

            print(f">>> Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")

        if self.wandb_logger is not None:
            self.wandb_logger.log(performance)

        return  ndcg_at_k(self.valid_gt_ids, total_top_k_document, 10)

    @torch.no_grad()
    def inference(self, device, zero_shot):
        self.model.to(device)
        self.model.eval()

        total_top_k_document = []
        for _, batch in enumerate(tqdm(self.test_dataloader, desc=f"Testing")):
            _, _, input_u = batch
            input_u = {key: val.to(device) if val is not None else val for key, val in input_u.items()}
            with autocast(dtype=torch.bfloat16, enabled=self.training_args['bf16'], device_type=device):
                output_u = self.model(**input_u)
                similarity_scores = self._ranking_predict(output_u, self.training_args, device)

            batch_top_k_indices = similarity_scores.topk(k=max(self.training_args['cutoff']), dim=-1).indices.tolist()

            for top_k_indices in batch_top_k_indices:
                total_top_k_document.append(top_k_indices)

        print(">> Test results:")
        performance = {}
        for k in self.training_args['cutoff']:
            recall = recall_at_k(self.test_gt_ids, total_top_k_document, k)
            ndcg = ndcg_at_k(self.test_gt_ids, total_top_k_document, k)
            if zero_shot:
                performance[f'zero_shot/Recall@{k}'] = recall
                performance[f'zero_shot/NDCG@{k}'] = ndcg
            else:
                performance[f'test/Recall@{k}'] = recall
                performance[f'test/NDCG@{k}'] = ndcg

            print(f">>> Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")

        recommendation_results = [{'gt': self.test_gt_ids[idx], 'recommendation': total_top_k_document[idx]}
                                  for idx in range(len(self.test_gt_ids))]

        if self.wandb_logger is not None:
            self.wandb_logger.log(performance)

        return performance, recommendation_results

    def train(self, ckpt_save_path, device):

        optimizer = self._set_optimizer(self.training_args['learning_rate'])
        scheduler = self._set_scheduler(optimizer=optimizer)
        epochs = self.training_args['epochs']

        self.model.to(device)
        print("\n> Zero-shot Performance.")
        self.item_embeddings = self._encode_items(self.item_dataloader, device=device).to(device)
        performance, recommendation_results  = self.inference(device=device, zero_shot=True)

        if self.training_args['zero_shot']:
            return performance, recommendation_results

        best_val_ndcg_at_k = -np.inf
        patience = 0
        for epoch in range(epochs):
            self.train_one_epoch(epoch, optimizer, scheduler, device=device)
            avg_val_ndcg_at_k = self.valid_one_epoch(epoch, device=device)

            if best_val_ndcg_at_k <= avg_val_ndcg_at_k:
                best_val_ndcg_at_k = avg_val_ndcg_at_k
                patience = 0
                self.model.save_checkpoint(ckpt_save_path=ckpt_save_path)
                print(f">> Best model saved with NDCG@10: {best_val_ndcg_at_k:.4f}")
            else:
                patience += 1

            if patience == self.training_args['patience']:
                print(f">> Early stopped after {epoch + 1} epochs.")
                break

        self.model.load_best_checkpoint(ckpt_save_path=ckpt_save_path)
        performance, recommendation_results = self.inference(device=device, zero_shot=False)

        return performance, recommendation_results
