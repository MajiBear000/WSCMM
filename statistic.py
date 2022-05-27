# -*- conding: utf-8 -*-
import os
from tqdm import tqdm
from data_loader import read_vua
from basic_extract import count_missing_basic, target_extract
from utils import load_json
import matplotlib.pyplot as plt
import csv

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

def save_tsv(data, path, headline=None):
    print(f'{path} len: {len(data)}')
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        if headline:
            writer.writerow(headline)
        writer.writerows(data)
        
def load_tsv(path):
    data=[]
    with open(path) as f:
        lines = csv.reader(f, delimiter='\t')
        next(lines)
        for line in lines:
            data.append(list(line))
    return data

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

def main():
    data = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    path = 'data/VUA18'
    val_path = 'data/VUA18/val.tsv'
    test_path = 'data/VUA20/test.tsv'
    train_path = 'data/VUA20/train.tsv'
    data_emb_path = 'data/VUA20/basic_emb.json'

    print('*****Load VUA Data*****')
    raw_train = load_tsv(train_path)
    raw_test = load_tsv(test_path)
    raw_val = load_tsv(val_path)
    data['train'] = vua_reform(raw_train, new=True)
    data['test'] = vua_reform(raw_test, new=True)
    data['val'] = vua_reform(raw_val)

    print('*****Filt Unkown Data*****')
    basic_train = load_json(data_emb_path)
    
    train_kn = filt_unk(basic_train, data['train'])
    save_tsv(train_kn, train_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])
    test_kn = filt_unk(basic_train, data['test'])
    save_tsv(test_kn, test_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])
    val_kn = filt_unk(basic_train, data['val'])
    save_tsv(val_kn, val_path.replace('.tsv', '_kn.tsv'), ['index','label','sentence','POS','FGPOS','w_index'])

    print('*****Count Missing Basic Mean*****')
    count_missing_basic(data)
    count_missing_basic({'train':data['train'], 'test':data['val']})
    count_missing_basic({'train':data['train'], 'test':data['train']})

    print('*****Extract Target Mean*****')
    data['train_basic'] = target_extract(data['train'])
    data['test_basic'] = target_extract(data['test'], basic=False)
    data['val_basic'] = target_extract(data['val'], basic=False)

    print('*****Trainset Missing Target Log*****')
    log_target_distribute(target_extract(data['train']))

if __name__ == '__main__':
    main()
