import json
import random
import numpy as np
import torch

def set_randomness(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return [json.loads(line) for line in f]


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)


def format_document(doc_info, data_name, used_info):
    doc = ''
    if 'm' in used_info:
        if 'pearl' in data_name:
            doc += (f"Title: {doc_info['title']}. Genre: {doc_info['genre']}. Cast:"
                    f" {', '.join(doc_info['cast'])}. "
                    f"Director: {', '.join(doc_info['director'])}\n")
        elif 'inspired' in data_name:
            doc += (
                f"Title: {doc_info['title']}. Genre: {doc_info['genre']}. Cast: {', '.join(doc_info['cast'])}. "
                f"Director: {', '.join(doc_info['director'])}\n")
        elif 'redial' in data_name:
            doc += (
                f"Title: {doc_info['title']}. Genre: {doc_info['genre']}. Cast: {', '.join(doc_info['cast'])}. "
                f"Director: {', '.join(doc_info['director'])}\n")
        else:
            raise ValueError(f"Invalid data_name in formatting document: {data_name}")
    if 'pref' in used_info:
        doc += ', '.join(doc_info['like'])
        doc += ', '.join(doc_info['dislike'])

    return doc


def load_document(file_path, data_name, used_info=None):
    document = load_json(file_path)
    item_ids = list(document.keys())
    document_dict = {'item': [format_document(document[_id], data_name, used_info) for _id in item_ids]}
    return item_ids, document_dict


def format_query(query, used_info):
    query_str = ''
    if 'c' in used_info:
        query_str += f"{query['input_dialog_history']}"
    elif 'l' in used_info:
        query_str += f"Like Preference: {','.join(query['preference']['like'])}"
    elif 'd' in used_info:
        query_str += f"Dislike Preferences: {','.join(query['preference']['dislike'])}"
    else:
        raise ValueError(f"Invalid used_info in formatting query: {used_info}")

    return query_str


def load_query(file_path, used_info):
    queries = load_jsonl(file_path)

    if 'test' in str(file_path):
        gt_ids = []
        for input_ in queries:
            gt_ids.append([gt['id'] for gt in input_['gt']]) # gt['id']: list of items
        conv_gt_ids = None

    elif 'train' in str(file_path):
        gt_ids = [input_['train_gt']['id'] for input_ in queries] # input_['train_gt']['id']: gt['id']: items
        conv_gt_ids = []
        for input_ in queries:
            conv_gt_ids.append([gt['id'] for gt in input_['gt']])
    else:
        gt_ids = [input_['gt']['id'] for input_ in queries] # input_['gt']['id']: gt['id']: items
        conv_gt_ids = None
        

    query_dict = {}
    for key in used_info:
        if key not in ['c', 'l', 'd']:
            raise ValueError(f"Invalid used_info in preparing query: {used_info}")
        query_dict[key] = [format_query(input_, key) for input_ in queries]

    return gt_ids, conv_gt_ids, query_dict

