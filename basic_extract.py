# -*- conding: utf-8 -*-
import os
from os.path import exists
import numpy as np
import torch
from tqdm import tqdm
from utils import tokenize_by_index, save_json, load_json, save_pth, load_pth

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
        basic_train[target]['idx']=[]
        for sentence, index in basic_train[target]['sam']:
            basic_train[target]['seq'].append(sentence)
            _, n_idx = tokenize_by_index(tokenizer, sentence, index)
            basic_train[target]['idx'].append(n_idx)
            '''
            if 'idx' not in basic_train[target].keys():
                basic_train[target]['idx'] = np.mat(n_idx)
            else:
                basic_train[target]['idx'] = np.vstack((basic_train[target]['idx'],np.array(n_idx)))
            '''
    print('Start generate basic mean embedding...')
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
                
        basic_emb[target] = vec.cpu().numpy().tolist()
    print('finished!')
        
    return basic_emb

def prepare_embedding(args, model, tokenizer, data, val=False):
    basic_emb_path = os.path.join(args.trainset_dir, 'basic_emb.json')

    if not exists(basic_emb_path):
        basic_train = target_extract(data['train'])
        basic_emb = basic_embedding(model, tokenizer, basic_train)
        save_json(basic_emb, basic_emb_path)
    else:
        basic_emb = load_json(basic_emb_path)
        print('Succeed load basic mean embedding:',len(basic_emb))
    
    data_emb_path = os.path.join(args.testset_dir, 'data_emb.pth') if val else os.path.join(args.trainset_dir, 'data_emb.pth')
    if exists(data_emb_path):
        data_emb = load_pth(data_emb_path)
        return data_emb

    data_emb = []
    print('Start output test embedding...')
    context_data = data['test'] if val else data['train']
    for sample in tqdm(context_data):
        target = sample[0]
        sentence = sample[1].lower()
        index = sample[2]
        label = torch.tensor(int(sample[3]))

        tokenized = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
        _, ni = tokenize_by_index(tokenizer, sentence, index)
        with torch.no_grad():
            outputs = model(**tokenized)
        con_vec = outputs[0][0][ni[0]]
        
        if target in basic_emb.keys():
            vec = np.array(basic_emb[target])
            basic_vec = torch.from_numpy(vec)
        else:
            tokenized = tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokenized)
            basic_vec = outputs[0][0][0]
            
        data_emb.append([torch.tensor(basic_vec), torch.tensor(con_vec), label])
    save_pth(data_emb, data_emb_path)
    print('test emb loaded!')
    return data_emb

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
