# -*- conding: utf-8 -*-
from utils import tokenize_by_index

def target_extract(train_set):
    basic_train = {}
    for sample in train_set:
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        label = str(sample[3])
        if label == '1':
            continue
        if target in basic_train.keys():
            basic_train[target].append([sentence, index])
        else:
            basic_train[target] = [[sentence, index]]

    return basic_train

def basic_embedding(model, tokenizer, basic_set):

    return 0

def count_missing_basic(model, tokenizer, data):
    basic_train = target_extract(data['train'])
    basic_test = target_extract(data['test'])
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

def prepare_embedding(model, tokenizer, data):
    basic_train = target_extract(data['train'])
    test_embedding = []
    for sample in data['train']:
        check_index(sample[1], sample[2], sample[0])
    for sample in data['test']:
        target = sample[0]
        sentence = sample[1]
        index = sample[2]
        label = str(sample[3])
        if target in basic_train.keys():
            basic_embedding(model, tokenizer, basic_train[target])
        else:
            embedding = tokenizer(sentence)
            token_ids, ni = tokenize_by_index(tokenizer, sentence, index)
            if not embedding['input_ids'] == token_ids:
                print(embedding['input_ids'])
                print(token_ids)
            print("-------------------")
            print(target)
            t_ids = token_ids[ni[0]]
            print(ni)
            print(tokenizer.decode(t_ids))
            
            

