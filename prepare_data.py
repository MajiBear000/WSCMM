from nltk.corpus import wordnet as wn
import spacy
from basic_extract import target_extract
from utils import tokenize_by_index
import torch
from tqdm import tqdm
import numpy as np
import random

def get_ids(args, tokenizer, raw_data):
    basic_train = target_extract(raw_data['train'])

    tokenized_train = prepare_ids(args, tokenizer, basic_train, raw_data['train'])
    tokenized_test = prepare_ids(args, tokenizer, basic_train, raw_data['test'])
    #tokenized_val = prepare_ids(args, tokenizer, basic_train, raw_data['val'])

    return {'train':tokenized_train, 'test':tokenized_test, 'val':tokenized_test}

def get_embs(args, roberta, tokenizer, data):
    train_emb = prepare_embedding(args, roberta, tokenizer, data, 'train')
    if args.ori_emb:
        test_emb = prepare_embedding(args, roberta, tokenizer, data, 'test')
        val_emb = prepare_embedding(args, roberta, tokenizer, data, 'val')
    else:
        test_emb = prepare_embedding(args, roberta, tokenizer, data, 'test_kn')
        val_emb = prepare_embedding(args, roberta, tokenizer, data, 'val_kn')
    return {'train':train_emb, 'test':test_emb, 'val':val_emb}

def prepare_ids(args, tokenizer, basic_train, data):
    tokenized_input = []
    skip_words = []
    basicer = DefaultBasic()
    for sample in tqdm(data):
        target = sample[0]
        sentence = sample[1].lower()
        index = sample[2]
        label = torch.tensor(int(sample[3]))
        
        con_token = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
        _, con_ni = tokenize_by_index(tokenizer, sentence, index)
        con_mask = torch.zeros(con_token['input_ids'][0].shape)
        con_mask[con_ni[0]]=1
        #print(con_token)

        if target in basic_train.keys():
            rand = random.randint(1,len(basic_train[target]['sam']))-1
            basic_sen, basic_idx = basic_train[target]['sam'][rand]
            
            basic_token = tokenizer(basic_sen,padding=True,truncation=True,max_length=512,return_tensors="pt")
            _, basic_ni = tokenize_by_index(tokenizer, basic_sen, basic_idx)
            basic_mask = torch.zeros(basic_token['input_ids'][0].shape)
            basic_mask[basic_ni[0]]=1
            
        else:
            if target in ['staying-on', 'A-levels', 'semi-skilled']:
                skip_words.append([target, sentence, label, index])
                continue
            basic_sen, basic_idx = basicer(target)
            if basic_sen == None:  
                skip_words.append([target, sentence, label, index])
                continue
            basic_sen = ' '.join(basic_sen)
            basic_token = tokenizer(basic_sen,padding=True,truncation=True,max_length=512,return_tensors="pt")
            _, basic_ni = tokenize_by_index(tokenizer, basic_sen, basic_idx)
            basic_mask = torch.zeros(basic_token['input_ids'][0].shape)
            basic_mask[basic_ni[0]]=1
        '''
        if (len(tokenized_input)) > 64:
            break
        '''
        tokenized_input.append([basic_token['input_ids'][0], basic_token['attention_mask'][0],
                                basic_mask, con_token['input_ids'][0], con_token['attention_mask'][0], con_mask, label])
    log_skip_words(skip_words)
    print(f"=============================Finished length: {len(tokenized_input)}!=============================")
    return tokenized_input

class DefaultBasic:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

    def __call__(self, word):
        sent = None
        index = None
        if len(wn.synsets(word))==0:
            return None, None
        syn = wn.synsets(word)[0]
        lemmas = syn.lemmas()
        examples = syn.examples()
        if not examples: # Means there is no example sentence in wordnet
            return None, None
        doc = self.nlp(examples[0])
        for lemma in lemmas:
            for idx, t in enumerate(doc):
                if lemma.name() == t.lemma_:
                    index = idx
                    sent = [token.text for token in doc]
                    break
            else:
                continue
            break
        return sent, index

def log_skip_words(skip_words):
    path = str(len(skip_words))+'skip_words.txt'
    with open(path, 'w') as f:   
        for data in skip_words:
            f.write(str(data[2]))
            f.write(' ')
            f.write(data[0])
            f.write(':  ')
            f.write(data[1])
            f.write(str(data[3]))
            f.write('\n') 

#if __name__ == '__main__':
#    basicer = DefaultBasic()
#    print(basicer('work'))
