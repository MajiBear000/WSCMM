# -*- conding: utf-8 -*-
# transformers version 2.11.0
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaModel
#from transformers.models.bert.modeling_bert import RobertaModel

# 1
def read_config(path):
    config_path = os.path.join(path, 'config.json')
    config = load_json(config_path)
    print(config)
    return config

# 2
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

# 3
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

# 4
def tokenize_by_index(tokenizer, seq, index=None, no_flat=False):
    seq = seq.split(' ')   # seq already being splited
    tokens_ids = [[tokenizer.bos_token_id]]
    for i,ele in enumerate(seq):
        if i:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele, add_prefix_space = True)))
        else:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele)))
    tokens_ids.append([tokenizer.eos_token_id])

    if not index==None:
        i_s = 0     #start index of target word
        for i, ele in enumerate(tokens_ids):
            i_e = i_s+len(ele)    #end index of target word
            if i == index+1:
                if not no_flat:
                    tokens_ids = sum(tokens_ids, [])  # return a flat ids list
                return tokens_ids, [i_s, i_e]
            i_s += len(ele)
    
    if not no_flat:
        tokens_ids = sum(tokens_ids, [])  # return a flat ids list
    return tokens_ids

# 5
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

# 6
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# 7
def save_pth(data, path):
    torch.save(data, path) #use tensor.clone() save pure data without relation

# 8
def load_pth(path):
    data = torch.load(path)
    return data

# 9
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# 10
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# 11
def compute_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    result = {
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "acc": acc,
    }
    return result

# 12
def log_results(results, val=False):
    if val:
        print("*****Validation results*****")
    else:
        print("*****Testing results*****")
    for key in results.keys():
        print(f" {key} = {results[key]}")

# 13
def output_param(model):
    for param in model.parameters():
        print(param.data)
        print(type(param.data), param.size())

# 14
def loss_plot(args, train_loss, val_loss):
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Dev loss')

    plt.title('Change in Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plot_path = os.path.join(args.plot_dir, args.stamp+'_loss.png') 
    plt.savefig(plot_path)

# 15
def acc_plot(args, pre, rec, f1, acc):
    plt.plot(pre, label='Precision')
    plt.plot(rec, label='Recall')
    plt.plot(f1, label='F1-score')
    plt.plot(acc, label='Accuracy')

    plt.title('Validation Metrics Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('%')

    plt.legend()
    plot_path = os.path.join(args.plot_dir, args.stamp+'_metrix.png') 
    plt.savefig(plot_path)

# 16
def test_metrix_log(path, result):
    with open(path, 'a+') as f:
        f.write('\n')
        for key in result:
            f.write(key)
            f.write(':  ')
            f.write(str(result[key]))
            f.write('\n')         

# 17
def check_index(sent, index, word):
    seq = sent.split(' ')
    iw = seq[index]
    if not iw==word:
        print("====incorrect index===")



    
