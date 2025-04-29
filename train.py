import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from wonderwords import RandomWord

from src import CoralTrainer, CoralDataset, CoralItemDataset, CoralEncoder
from utils import load_query, load_document, set_randomness


def main(training_args):
    training_args = vars(training_args)
    print("> Training arguments")
    for k, v in training_args.items():
        print(f'{k}: {v}')

    data_path = Path('data') / training_args['data_name']
    train_path = data_path / 'input_processed_train.jsonl'
    valid_path = data_path / 'input_processed_valid.jsonl'
    test_path = data_path / 'input_processed_test.jsonl'
    document_path = data_path / 'processed_document.json'

    train_gt_ids, train_conv_gt_ids, train_u = load_query(file_path=train_path,
                                                          used_info=training_args['query_used_info'])
    valid_gt_ids, _, valid_u = load_query(file_path=valid_path,
                                          used_info=training_args['query_used_info'])
    test_gt_ids, _, test_u = load_query(file_path=test_path,
                                        used_info=training_args['query_used_info'])
    document_ids, document = load_document(file_path=document_path, data_name=training_args['data_name'],
                                           used_info=training_args['doc_used_info'])
    # convert item(e.g., 'Hoffa (1992)' id to idx (e.g., 1628)
    idx2item = document_ids
    item2idx = {v: k for k, v in enumerate(idx2item)}
    for idx, cgt in enumerate(train_conv_gt_ids):
        train_conv_gt_ids[idx] = [item2idx[i] for i in cgt]
    train_gt_ids = [item2idx[i] for i in train_gt_ids]
    valid_gt_ids = [item2idx[i] for i in valid_gt_ids]
    test_gt_ids = [[item2idx[i] for i in test] for test in test_gt_ids]

    print("\n\n> Sample data:")
    if 'c' in training_args['query_used_info']:
        print(train_u['c'][0])
    if 'l' in training_args['query_used_info']:
        print(train_u['l'][0])
    if 'd' in training_args['query_used_info']:
        print(train_u['d'][0])


    train_dataset = CoralDataset(user_text=train_u, gt_ids=train_gt_ids,
                                 base_tokenizer_name=training_args['base_model_name'])
    valid_dataset = CoralDataset(user_text=valid_u, gt_ids=valid_gt_ids,
                                 base_tokenizer_name=training_args['base_model_name'])
    test_dataset = CoralDataset(user_text=test_u, gt_ids=test_gt_ids,
                                base_tokenizer_name=training_args['base_model_name'])
    item_dataset = CoralItemDataset(item_text=document, base_tokenizer_name=training_args['base_model_name'])

    train_dataloader = DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2 * training_args['batch_size'], shuffle=False,
                                  collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=2 * training_args['batch_size'], shuffle=False,
                                 collate_fn=test_dataset.collate_fn)
    train_infer_dataloader = DataLoader(train_dataset, batch_size=4 * training_args['batch_size'], shuffle=False,
                                        collate_fn=train_dataset.collate_fn)
    item_dataloader = DataLoader(item_dataset, batch_size=4 * training_args['batch_size'], shuffle=False,
                                 collate_fn=item_dataset.collate_fn)

    random_word_generator = RandomWord()
    while True:
        random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]
        if " " in random_word or "-" in random_word:
            continue
        else:
            break
    random_word_and_date = random_word + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if training_args['wandb_project'] is not None:
        wandb_tag = [
            training_args['data_name'],
            f"seed_{args.seed}",
            f"query_used_info_{training_args['query_used_info']}",
            f"doc_used_info_{training_args['doc_used_info']}",
        ]
        wandb_logger = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=random_word_and_date,
            group=args.wandb_group if args.wandb_group else training_args['data_name'],
            config=training_args,
            tags=wandb_tag
        )
    else:
        wandb_logger = None

    model = CoralEncoder(base_model_name=training_args['base_model_name'],
                         alpha=training_args['alpha'],
                         beta=training_args['beta'])

    trainer = CoralTrainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        train_infer_dataloader=train_infer_dataloader,
        item_dataloader=item_dataloader,
        item_num=len(document_ids),
        train_conv_gt_ids=train_conv_gt_ids,
        train_gt_ids=train_gt_ids,
        valid_gt_ids=valid_gt_ids,
        test_gt_ids=test_gt_ids,
        optimizer='Adam',
        scheduler='get_linear_schedule_with_warmup',
        accumulation_steps=training_args['accumulation_steps'],
        training_args=training_args,
        wandb_logger=wandb_logger
    )

    ckpt_save_path = os.path.join(training_args['ckpt_save_path'], training_args['data_name'],
                                  random_word_and_date)
    try:
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
    except OSError:
        print("Error: Failed to create the directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    performance, recommendation_results = trainer.train(ckpt_save_path, device=device)

    output_results_path = Path('outputs') / 'results'
    os.makedirs(output_results_path, exist_ok=True)

    with open(os.path.join(output_results_path, f'{random_word_and_date}.json'), 'w') as f:
        json.dump({
            'version': random_word_and_date,
            'arguments': training_args,
            'performance': performance,
            'recommendation_results': recommendation_results
        }, f, indent=1, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--zero_shot', action='store_true')
    # data
    parser.add_argument('--data_name', type=str, required=True, choices=['inspired', 'redial', 'pearl'])
    # hyperparameter
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument("--query_used_info", type=str, nargs='+', choices=['c', 'l', 'd'], default=['c', 'l', 'd'])
    parser.add_argument("--doc_used_info", type=str, nargs='+', choices=['m', 'pref'], default=['m', 'pref'])
    # train
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--negative_sample", type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--base_model_name', type=str, default='nvidia/NV-Embed-v1')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ckpt_save_path', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--cutoff', type=int, nargs='+', default=[5, 10, 50])
    # wandb
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument('--wandb_group', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)

    main(args)
