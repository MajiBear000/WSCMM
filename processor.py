# -*- conding: utf-8 -*-
import os
import random
import torch
import logging

from os.path import exists
from tqdm import tqdm
import numpy as np
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, IterableDataset
from torch.nn.utils.rnn import pad_sequence

from basic_extract import target_extract, DefaultBasic
from sw.utils import tokenize_by_index, save_json, load_json, save_pth, load_pth

logger = logging.getLogger(__name__)

class prepare_data(object):
    def __init__(self, args, raw_data):
        self.data = {'train':None, 'test':None, 'val':None}
        self.args = args
        self.raw_data = raw_data
        self.basic = target_extract(raw_data['train'])

    def get_ids(self, tokenizer):
        self.data['train'] = BasicIdsProcessor(self.args, tokenizer, self.basic, self.raw_data, 'train')
        self._log_skip_words(os.path.join(self.args.trainset_dir, 'train_'), self.data['train'].skip_words)
        self.data['test'] = BasicIdsProcessor(self.args, tokenizer, self.basic, self.raw_data, 'test')
        self._log_skip_words(os.path.join(self.args.trainset_dir, 'test_'), self.data['test'].skip_words)
        #val = BasicIdsProcessor(args, tokenizer, basic_train, raw_data['val'])

        return self.data

    def get_embs(self, roberta, tokenizer):
        '''
        train_emb = prepare_embedding(args, roberta, tokenizer, raw_data, 'train')
        if args.ori_emb:
            test_emb = prepare_embedding(args, roberta, tokenizer, raw_data, 'test')
            val_emb = prepare_embedding(args, roberta, tokenizer, raw_data, 'val')
        else:
            test_emb = prepare_embedding(args, roberta, tokenizer, raw_data, 'test_kn')
            val_emb = prepare_embedding(args, roberta, tokenizer, raw_data, 'val_kn')
        '''
        
        self.data['train'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'train')
        self.data['test'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'test')
        self.data['val'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'val')
        return {'train':train_emb, 'test':test_emb, 'val':val_emb}
    
    def _log_skip_words(self, path, skip_words):
        path += str(len(skip_words))+'_skipped.txt'
        with open(path, 'w') as f:   
            for data in skip_words:
                f.write(str(data[2]))
                f.write(' ')
                f.write(data[0])
                f.write(':  ')
                f.write(data[1])
                f.write(str(data[3]))
                f.write('\n') 


class Processor(IterableDataset):
    def __init__(self, args, tokenizer, raw_data, type_='test'):
        self.args = args
        self.tokenizer = tokenizer
        self.raw_data = raw_data[type_]
        self.skip_words = []
        self._batch_size = args.train_batch_size if type_=='train' else args.test_batch_size

    def __iter__(self):
        raise ValueError("No __iter__ func in Processor has been declared!")

    def __len__(self):
        raise ValueError("No __len__ func in Processor has been declared!")

    def _return_sampler(self, type_='test'):
        if type_ == 'train':
            return RandomSampler
        else:
            return SequentialSampler


class BasicEmbsProcessor(Processor):
    '''
        Process and Load data for ClassificationForBasicMean_Linear Model
        args.(trainset/teatset/valset)_dir : 
        args.model_name : name of model
        args.device : index of which device used
        args.ori_emb : True if contain unknown words
    '''
    def __init__(self, args, model, tokenizer, basic_train, raw_data, type_='test', basicer=DefaultBasic()):
        super(BasicEmbsProcessor, self).__init__(args, tokenizer, raw_data, type_)
        self.model = model
        self.basic_train = basic_train
        self._basicer = basicer
        self._basic_emb = {}
        self.emb_input = []

        self._basic_embedding()
        self._emb_path = self._trace_data(type_)
        self._prepare_embs(type_)
        self._sampler = self._return_sampler(type_)
        self._batch_sampler = BatchSampler(self._sampler(self.tokenized_input), self._batch_size, False)
        self._max_steps = len(self._batch_sampler)

    def _basic_embedding(self):
        _basic_emb_path = os.path.join(self.args.trainset_dir, 'basic_emb.json')

        if not exists(_basic_emb_path):
            for target in tqdm(self.basic_train.keys()):
                self.basic_train[target]['seq']=[]
                self.basic_train[target]['idx']=[]
                for sentence, index in self.basic_train[target]['sam']:
                    self.basic_train[target]['seq'].append(sentence)
                    _, ni = tokenize_by_index(self.tokenizer, sentence, index)
                    self.basic_train[target]['idx'].append(ni)
            logger.info('Start generate basic mean embedding...')
            for target in tqdm(self.basic_train.keys()):
                t_size = len(self.basic_train[target]['seq'])
                for i, seq in enumerate(self.basic_train[target]['seq']):
                    tokenized = self.tokenizer(seq,padding=True,truncation=True,max_length=512,return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**tokenized)
                
                    if i ==0 :
                        vec = outputs[0][0][self.basic_train[target]['idx'][0][0]]/t_size
                    else:
                        vec += outputs[0][0][self.basic_train[target]['idx'][i][0]]/t_size #[last_hidden_state][batch][index][768]
                
                self._basic_emb[target] = vec.cpu().numpy().tolist()
            logger.info('basic mean embedding Generated!')
            save_json(self._basic_emb, _basic_emb_path)
        else:
            self._basic_emb = load_json(_basic_emb_path)
            logger.info('Succeed load basic mean embedding:',len(self._basic_emb))

    def _trace_data(opt):
        if opt is None:
            logger.warning("ERROR: give a dataset you want to trace")
            exit()
        elif opt == 'train':
            path = os.path.join(args.trainset_dir, 'data_emb.pth')
        elif opt == 'test':
            path = os.path.join(args.testset_dir, 'test_emb.pth')
        elif opt == 'val':
            path = os.path.join(args.valset_dir, 'val_emb.pth')
        
        elif opt == 'test_kn':      
            path = os.path.join(args.testset_dir, 'test_emb_kn.pth')
        elif opt == 'val_kn':      
            path = os.path.join(args.valset_dir, 'val_emb_kn.pth')
        return path

    def _prepare_embs(self, opt):
        if exists(self._emb_path):
            self.emb_input = load_pth(self._emb_path)
            logger.info(f'data emb loaded: {len(data_emb)}')

        logger.info('Start load input embedding...')
        for sample in tqdm(self.raw_data):
            target = sample[0]
            sentence = sample[1].lower()
            index = sample[2]
            label = torch.tensor(int(sample[3]))

            tokenized = self.tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
            _, ni = tokenize_by_index(self.tokenizer, sentence, index)
            with torch.no_grad():
                outputs = self.model(**tokenized)
            con_vec = outputs[0][0][ni[0]]
        
            if target in basic_emb.keys():
                vec = np.array(basic_emb[target])
                basic_vec = torch.from_numpy(vec)
            elif opt in ['train', 'test', 'val']:
                tokenized = tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**tokenized)
                basic_vec = outputs[0][0][0]
            else:
                continue
            
            self.emb_input.append([basic_vec, con_vec, label])
        save_pth(self.emb_input, self._emb_path)
        logger.info(f'data emb loaded and saved: {len(data_emb)}')

    def _sample_batch(self, idxs):
        batch = []
        for i in idxs:
            sample = self.emb_input[i]
            batch.append(sample)
        return zip(*batch)
    
    def __iter__(self):
        for idxs in self._batch_sampler:
            batch = self._sample_batch(idxs)
            batch = tuple(t.to(self.args.device) for t in batch)
            yield batch

    def __len__(self):
        return self._max_steps


class BasicIdsProcessor(Processor):
    '''
        Process and Load data for ClassificationForBasicMean_Roberta Model
        args.model_name : name of model
        args.device : index of which device used
        args.ori_emb : True if contain unknown words
    '''
    def __init__(self, args, tokenizer, basic_train, raw_data, type_='test', basicer=DefaultBasic()):
        super(BasicIdsProcessor, self).__init__(args, tokenizer, raw_data, type_)
        self.basic_train = basic_train
        self._basicer = basicer
        self.tokenized_input = []
        
        self._prepare_ids()
        self._sampler = self._return_sampler(type_)
        self._batch_sampler = BatchSampler(self._sampler(self.tokenized_input), self._batch_size, False)
        self._max_steps = len(self._batch_sampler)

    def _prepare_ids(self):
        for sample in tqdm(self.raw_data):
            target = sample[0]
            sentence = sample[1].lower()
            index = sample[2]
            label = torch.tensor(int(sample[3]))
        
            con_token = self.tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
            _, con_ni = tokenize_by_index(self.tokenizer, sentence, index)
            con_mask = torch.zeros(con_token['input_ids'][0].shape)
            con_mask[con_ni[0]]=1
            #print(con_token)

            if target in self.basic_train.keys():
                rand = random.randint(1,len(self.basic_train[target]['sam']))-1
                basic_sen, basic_idx = self.basic_train[target]['sam'][rand]
            
                basic_token = self.tokenizer(basic_sen,padding=True,truncation=True,max_length=512,return_tensors="pt")
                _, basic_ni = tokenize_by_index(self.tokenizer, basic_sen, basic_idx)
                basic_mask = torch.zeros(basic_token['input_ids'][0].shape)
                basic_mask[basic_ni[0]]=1
            elif not self.args.ori_emb:
                continue                
            else:
                basic_sen, basic_idx = self._basicer(target)
                if basic_sen == None:  
                    self.skip_words.append([target, sentence, label, index])
                    continue
                basic_sen = ' '.join(basic_sen)
                basic_token = self.tokenizer(basic_sen,padding=True,truncation=True,max_length=512,return_tensors="pt")
                _, basic_ni = tokenize_by_index(self.tokenizer, basic_sen, basic_idx)
                basic_mask = torch.zeros(basic_token['input_ids'][0].shape)
                basic_mask[basic_ni[0]]=1
            # break if (len(self.tokenized_input)) > 64   
            self.tokenized_input.append([basic_token['input_ids'][0],basic_token['attention_mask'][0],
                                    basic_mask, con_token['input_ids'][0],con_token['attention_mask'][0],
                                    con_mask, label])
        logger.info(f"=====Finished length: {len(self.tokenized_input)}!=====")

    def _collate_fn(self, batch):
        basic_ids,basic_attention,basic_mask,con_ids,con_attention,con_mask,labels = zip(*batch)

        basic_ids = pad_sequence(basic_ids, batch_first=True, padding_value=0)
        basic_attention = pad_sequence(basic_attention, batch_first=True, padding_value=0)
        basic_mask = pad_sequence(basic_mask, batch_first=True, padding_value=0)
        con_ids = pad_sequence(con_ids, batch_first=True, padding_value=0)
        con_attention = pad_sequence(con_attention, batch_first=True, padding_value=0)
        con_mask = pad_sequence(con_mask, batch_first=True, padding_value=0)
        labels = torch.tensor([t for t in labels])

        return basic_ids,basic_attention,basic_mask,con_ids,con_attention,con_mask,labels

    def _sample_batch(self, idxs):
        batch = []
        for i in idxs:
            sample = self.tokenized_input[i]
            batch.append(sample)
        return self._collate_fn(batch)
    
    def __iter__(self):
        for idxs in self._batch_sampler:
            batch = self._sample_batch(idxs)
            batch = tuple(t.to(self.args.device) for t in batch)
            yield batch

    def __len__(self):
        return self._max_steps










        

    
