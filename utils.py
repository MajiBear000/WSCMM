# -*- conding: utf-8 -*-
# transformers version 2.11.0
import os
import json
import random
import numpy as np
import torch

from transformers import RobertaTokenizer, RobertaModel
#from transformers.models.bert.modeling_bert import RobertaModel

def read_config(path):
    config_path = os.path.join(path, 'config.json')
    config = load_json(config_path)
    print(config)
    return config

def get_model(path):
    model = None
    try:
        config = read_config(path)
    except:
        print('===========Fail to load config.json!============')
    if config['_name_or_path'] == 'roberta-base':
        model = RobertaModel.from_pretrained(path)

    if model == None:
        print('===========Fail to load Model!============')
    
    return model

def get_tokenizer(path):
    tokenizer = None
    try:
        config = read_config(path)
    except:
        print('===========Fail to load config.json!============')
    vocab_file = os.path.join(path, 'vocab.json')
    merges_file = os.path.join(path, 'merges.txt')

    if config['_name_or_path'] == 'roberta-base':
        tokenizer = RobertaTokenizer(vocab_file, merges_file)

    if tokenizer == None:
        print('===========Fail to load Tokenizer!============')

    return tokenizer

def tokenize_by_index(tokenizer, seq, index=None, no_flat=False):
    seq = seq.split()   # seq already being splited
    tokens_ids = [[tokenizer.bos_token_id]]
    for i,ele in enumerate(seq):
        if i:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele, add_prefix_space = True)))
        else:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele)))
    tokens_ids.append([tokenizer.eos_token_id])

    if not index==None:
        i_s = 0     #start index of target word
        for i, ele in enumerate(tokens_ids):
            i_e = i_s+len(ele)-1    #end index of target word
            if i == index+1:
                if not no_flat:
                    tokens_ids = sum(tokens_ids, [])  # return a flat ids list
                return tokens_ids, [i_s, i_e]
            i_s += len(ele)
    
    if not no_flat:
        tokens_ids = sum(tokens_ids, [])  # return a flat ids list
    return tokens_ids

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_pth(data, path):
    torch.save(data, path) #use tensor.clone() save pure data without relation

def load_pth(path):
    data = torch.load(path)
    return data

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    


