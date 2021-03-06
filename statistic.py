# -*- conding: utf-8 -*-
import os
import csv
import logging

import matplotlib.pyplot as plt
from tqdm import tqdm

from sw.utils import load_json, save_tsv, load_tsv
from data_loader import _read_vua
from basic_extract import count_missing_basic, target_extract

logger = logging.getLogger(__name__)

def filt_unk(basic_train, data):
    data_kn = []
    for sample in tqdm(data):
        index = sample[6]
        label = str(sample[3])
        sentence = sample[1]
        pos = sample[4]
        fgpos = sample[5]
        w_index = str(sample[2])
        target = sample[0]
        if target in basic_train.keys():
            data_kn.append([index, label, sentence, pos, fgpos, w_index])
    return data_kn

def vua_reform(data, new=False):
    r_data=[]
    for s in tqdm(data):
        sentence = s[2]
        if new:
            w_index = int(s[5])
            fgpos = s[4]
        else:
            w_index = int(s[4])
            fgpos = s[3]
        target = sentence.split()[w_index]
        label = int(s[1])
        index = s[0]
        pos = s[3]
        
        r_data.append([target, sentence, w_index, label, pos, fgpos, index])
    return r_data

def log_target_distribute(basic_train):
    data_len=[]
    for key in basic_train.keys():
        data_len.append(len(basic_train[key]['sam']))
        if len(basic_train[key]['sam'])>1000:
            print(f"{len(basic_train[key]['sam'])} : {key}")
    data_len.sort(reverse=True)
    plt.plot(data_len,label='num')
    plt.title('Sorted Basic Mean Sample Size')
    plt.xlabel('target')
    plt.ylabel('samples')
    plot_path = 'data/VUA20/basic_len.png'
    plt.savefig(plot_path)
    return 0

def check_tokenizer(data):
    count = 0
    for key in data.keys():
        for sample in data[key]:
            seq = sample[1]
            if not seq.split()==seq.split(' '):
                logger.info(seq)
                count += 1
    logger.info(f'********{count}***********')        

def main():
    data = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    path = 'data/VUA18'
    val_path = 'data/VUA18/val.tsv'
    test_path = 'data/VUA20/test.tsv'
    train_path = 'data/VUA20/train.tsv'
    data_emb_path = 'data/VUA20/basic_emb.json'

    logger.info('*****Load VUA Data*****')
    raw_train = load_tsv(train_path)
    raw_test = load_tsv(test_path)
    raw_val = load_tsv(val_path)
    data['train'] = vua_reform(raw_train, new=True)
    data['test'] = vua_reform(raw_test, new=True)
    data['val'] = vua_reform(raw_val)

    logger.info('***** Check Split Works *****')
    check_tokenizer(data)
    '''
    logger.info('*****Filt Unkown Data*****')
    basic_train = load_json(data_emb_path)
    
    train_kn = filt_unk(basic_train, data['train'])
    save_tsv(train_kn, train_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])
    test_kn = filt_unk(basic_train, data['test'])
    save_tsv(test_kn, test_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])
    val_kn = filt_unk(basic_train, data['val'])
    save_tsv(val_kn, val_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])

    logger.info('*****Count Missing Basic Mean*****')
    count_missing_basic(data)
    count_missing_basic({'train':data['train'], 'test':data['val']})
    count_missing_basic({'train':data['train'], 'test':data['train']})

    logger.info('*****Extract Target Mean*****')
    data['train_basic'] = target_extract(data['train'])
    data['test_basic'] = target_extract(data['test'], basic=False)
    data['val_basic'] = target_extract(data['val'], basic=False)

    logger.info('*****Trainset Missing Target Log*****')
    log_target_distribute(target_extract(data['train']))
    '''
if __name__ == '__main__':
    main()
