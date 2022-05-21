# -*- conding: utf-8 -*-
# transformers version 2.11.0
import os
import json

from transformers import RobertaTokenizer, RobertaModel
#from transformers.models.bert.modeling_bert import RobertaModel

def read_config(path):
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
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

def tokenize_by_index(tokenizer, seq):
    seq = seq.split() # seq already being splited
    tokens_ids = [[tokenizer.bos_token_id]]
    for i,ele in enumerate(seq):
        if i:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele, add_prefix_space = True)))
        else:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele)))
    tokens_ids.append([tokenizer.eos_token_id])
    return tokens_ids