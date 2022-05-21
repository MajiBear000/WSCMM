# -*- conding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from utils import tokenize_by_index

def target_extract(train_set, basic=True):
    basic_train = {}
    for sample in tqdm(train_set):
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        label = str(sample[3])
        if label == '1' and basic:
            continue
        if target in basic_train.keys():
            basic_train[target]['sam'].append([sentence, index])
        else:
            basic_train[target] = {'sam':[[sentence, index]]}

    return basic_train

def basic_embedding(model, tokenizer, basic_train):
    basic_emb = {}
    for target in tqdm(basic_train.keys()):
        basic_train[target]['seq']=[]
        for sentence, index in basic_train[target]['sam']:
            basic_train[target]['seq'].append(sentence)
            _, n_idx = tokenize_by_index(tokenizer, sentence, index)
            if 'idx' not in basic_train[target].keys():
                basic_train[target]['idx'] = np.mat(n_idx)
            else:
                basic_train[target]['idx'] = np.vstack((basic_train[target]['idx'],np.array(n_idx)))
            
    for target in tqdm(basic_train.keys()):
        basic_emb[target] = []
        t_size = len(basic_train[target]['seq'])
        for i, seq in enumerate(basic_train[target]['seq']):
            tokenized = tokenizer(seq,padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokenized)
                
            if i ==0 :
                vec = outputs[0][0][basic_train[target]['idx'][0][0]]/t_size
            else:
                vec += outputs[0][0][basic_train[target]['idx'][i][0]]/t_size #[last_hidden_state][batch][index][768]
                
        basic_emb[target] = vec         
        
    return basic_emb

def prepare_embedding(model, tokenizer, data):
    
    basic_train = target_extract(data['train'])
    
    for sample in data['train']:
        check_index(sample[1], sample[2], sample[0])
    basic_emb = basic_embedding(model, tokenizer, basic_train)

    test_emb = []
    for sample in data['test']:
        target = sample[0]
        sentence = sample[1].lower()
        index = sample[2]
        label = str(sample[3])

        tokenized = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
        _, ni = tokenize_by_index(tokenizer, sentence, index)
        with torch.no_grad():
            outputs = model(**tokenized)
        test_vec = outputs[0][0][ni[0]]
        
        if target in basic_emb.keys():
            target_vec = basic_emb[target]
        else:
            tokenized = tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokenized)
            target_vec = outputs[0][0][0]

        test_emb.append([target_vec, test_vec, label])
    return test_emb

def count_missing_basic(data):
    basic_train = target_extract(data['train'])
    basic_test = target_extract(data['test'], basic=False)
    count = 0
    for key in basic_test.keys():
        if key in basic_train.keys():
            continue
        count += 1
    print(f'num of missing basic means: {count}')
    return 0

def check_index(sent, index, word):
    seq = sent.split()
    iw = seq[index]
    if not iw==word:
        print("====incorrect index===")
