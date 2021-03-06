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
from torch.nn import functional as F

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
        self.data['train'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'train')
        if self.args.ori_emb:
            self.data['test'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'test')
            self.data['val'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'val')
        else:
            self.data['test'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'test_kn')
            self.data['val'] = BasicEmbsProcessor(self.args, roberta, tokenizer, self.basic, self.raw_data, 'val_kn')

        return self.data

    def melbert_ids(self, tokenizer):
        self.data['train'] = MelbertProcessor(self.args, tokenizer, self.raw_data, 'train')
        self.data['test'] = MelbertProcessor(self.args, tokenizer, self.raw_data, 'test')

        return self.data
    
    def melbert_ids(self, tokenizer, basic_tokenizer):
        self.data['train'] = BaMeBertProcessor(self.args, tokenizer, basic_tokenizer, self.basic, self.raw_data, 'train')
        #self._log_skip_words(os.path.join(self.args.trainset_dir, 'train_'), self.data['train'].skip_words)
        self.data['test'] = BaMeBertProcessor(self.args, tokenizer, basic_tokenizer, self.basic, self.raw_data, 'test')
        #self._log_skip_words(os.path.join(self.args.trainset_dir, 'test_'), self.data['test'].skip_words)

        return self.data
         
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
            logger.info(f'Succeed load basic mean embedding: {len(self._basic_emb)}')

    def _trace_data(self, opt):
        if opt is None:
            logger.warning("ERROR: give a dataset you want to trace")
            exit()
        elif opt == 'train':
            path = os.path.join(self.args.trainset_dir, 'data_emb.pth')
        elif opt == 'test':
            path = os.path.join(self.args.testset_dir, 'test_emb.pth')
        elif opt == 'val':
            path = os.path.join(self.args.valset_dir, 'val_emb.pth')
        
        elif opt == 'test_kn':      
            path = os.path.join(self.args.testset_dir, 'test_emb_kn.pth')
        elif opt == 'val_kn':      
            path = os.path.join(self.args.valset_dir, 'val_emb_kn.pth')
        return path

    def _prepare_embs(self, opt):
        if exists(self._emb_path):
            self.emb_input = load_pth(self._emb_path)
            logger.info(f'data emb loaded: {len(data_emb)}')
            return 0

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
        
            if target in self._basic_emb.keys():
                vec = np.array(self._basic_emb[target])
                basic_vec = torch.from_numpy(vec)
            elif opt in ['train', 'test', 'val']:
                tokenized = self.tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**tokenized)
                basic_vec = outputs[0][0][0]
            else:
                continue
            
            self.emb_input.append([basic_vec, con_vec, label])
        save_pth(self.emb_input, self._emb_path)
        logger.info(f'data emb loaded and saved: {len(self.emb_input)}')

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


class BaMeBertProcessor(Processor):
    '''
        Process and Load data for ClassificationForBasicMean_Roberta Model
        args.model_name : name of model
        args.device : index of which device used
        args.ori_emb : True if contain unknown words
    '''
    def __init__(self, args, tokenizer, basic_tokenizer, basic_train,, raw_data, type_='test', basicer=DefaultBasic()):
        super(BaMeBertProcessor, self).__init__(args, tokenizer, raw_data, type_)
        self.basicer = basicer
        self.basic_train = basic_train
        self.tokenized_input = []
        self.basic_tokenizer = basic_tokenizer
        
        self._prepare_ids()
        self._sampler = self._return_sampler(type_)
        self._batch_sampler = BatchSampler(self._sampler(self.tokenized_input), self._batch_size, False)
        self._max_steps = len(self._batch_sampler)

    def _prepare_ids(self):
        logger.info('************ MelBERT Data Processor ***************')
        for sample in tqdm(self.raw_data):
            target = sample[0]
            sentence = sample[1].lower()
            index = sample[2]
            label = torch.tensor(int(sample[3]))
            pos = sample[4]

            # --------------------Context encoding------------------------#
            con_tokens, con_mask, con_attention, token_type_ids = self._get_seq_ids(target, sentence, index, pos, self.tokenizer)

            # ------------------Isolated target encoding----------------------#
            # get tokens of isolated target #
            isolate_ids_attetion = self.tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
            isolate_tokens = self.tokenizer.convert_ids_to_tokens(isolate_ids_attetion['input_ids'][0])
            isolate_attention = isolate_ids_attetion['attention_mask'][0]
            isolate_mask = torch.zeros(isolate_attention.shape)
            isolate_mask[1:-1]=1

            # --------------------Basic Mean encoding------------------------#
            if target in self.basic_train.keys():
                rand = random.randint(1,len(self.basic_train[target]['sam']))-1
                basic_sentence, basic_index = self.basic_train[target]['sam'][rand]
                basic_tokens, basic_mask, basic_attention, basic_token_type_ids = self._get_seq_ids(target, basic_sentence, basic_index, pos, self.basic_tokenizer)
            elif not self.args.ori_emb:
                basic_tokens, basic_mask, basic_attention, basic_token_type_ids = self._get_seq_ids(target, sentence, index, pos, self.tokenizer)    
            else:
                basic_sen, basic_idx = self._basicer(target)
                if basic_sen == None:
                    basic_tokens, basic_mask, basic_attention, basic_token_type_ids = self._get_seq_ids(target, sentence, index, pos, self.tokenizer)
                basic_tokens, basic_mask, basic_attention, basic_token_type_ids = self._get_seq_ids(target, basic_sentence, basic_index, pos, self.basic_tokenizer)

            # --------------------Convert to ids------------------------#
            # convert tokens to ids #
            con_ids = self.tokenizer.convert_tokens_to_ids(con_tokens)
            con_ids = torch.tensor(con_ids, dtype=torch.long)
            basic_ids = self.basic_tokenizer.convert_tokens_to_ids(basic_tokens)
            basic_ids = torch.tensor(basic_ids, dtype=torch.long)
            isolate_ids = self.tokenizer.convert_tokens_to_ids(isolate_tokens)
            isolate_ids = torch.tensor(isolate_ids, dtype=torch.long)

            # change input type #
            token_type_ids = token_type_ids.int()
            basic_token_type_ids = token_type_ids.int()
            
            #if (len(self.tokenized_input)) > 64:
                #break
            self.tokenized_input.append([con_ids, isolate_ids, con_mask, isolate_mask,
                                         con_attention, isolate_attention, token_type_ids,
                                         basic_ids, basic_mask, basic_attention, basic_token_type_ids,
                                         label])
        logger.info(f"=====Finished length: {len(self.tokenized_input)}!=====")

    def _get_seq_ids(self, target, sentence, index, pos, tokenizer):
        # get tokens of context and target #
        _ids_attetion = tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(_ids_attetion['input_ids'][0])
        attention = _ids_attetion['attention_mask'][0]
        _, _ni = tokenize_by_index(tokenizer, sentence, index)
        mask = torch.zeros(attention.shape)
        mask[_ni[0]:_ni[1]]=1
        token_type_ids = mask
        if not tokenizer.convert_tokens_to_ids(tokens)==_:
            print('*********** Token Split Error! ***********')
            
        # POS tag adding #
        _pad_len=1
        if self.args.use_pos:
            _pad_len = len(tokens)
            _pos_token = tokenizer.tokenize(pos)
            tokens += _pos_token + [tokenizer.sep_token]
            _pad_len = len(tokens) - _pad_len
            _padding = (0, pad_len)
            attention = F.pad(attention, _padding, value=1)
            mask = F.pad(mask, _padding, value=0)
            token_type_ids = F.pad(token_type_ids, _padding, value=3)     

        # set local content #
        if self.args.use_local_context:
            _local_start = 1
            _local_end = len(_tokens) - _pad_len
            _comma1 = tokenizer.tokenize(",")[0]
            _comma2 = tokenizer.tokenize(" ,")[0]
            for i, w in enumerate(_tokens):
                if i < _ni[0] and (w in [_comma1, _comma2]):
                    _local_start = i+1
                if i > _ni[1]-1 and (w in [_comma1, _comma2]):
                    _local_end = i
                    break
            token_type_ids[local_start:local_end] = 2
            token_type_ids[_ni[0]:_ni[1]]=1
        else:
            token_type_ids[_ni[0]:_ni[1]]=1

        return tokens, mask, attention, token_type_ids

    def _collate_fn(self, batch):
        con_ids, isolate_ids, con_mask, isolate_mask, con_attention, isolate_attention, token_type_ids, basic_ids, basic_mask, basic_attention, basic_token_type_ids, labels = zip(*batch)

        con_ids = pad_sequence(con_ids, batch_first=True, padding_value=0)
        isolate_ids = pad_sequence(isolate_ids, batch_first=True, padding_value=0)
        con_mask = pad_sequence(con_mask, batch_first=True, padding_value=0)
        isolate_mask = pad_sequence(isolate_mask, batch_first=True, padding_value=0)
        con_attention = pad_sequence(con_attention, batch_first=True, padding_value=0)
        isolate_attention = pad_sequence(isolate_attention, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        
        basic_ids = pad_sequence(basic_ids, batch_first=True, padding_value=0)
        basic_mask = pad_sequence(basic_mask, batch_first=True, padding_value=0)
        basic_attention = pad_sequence(basic_attention, batch_first=True, padding_value=0)
        basic_token_type_ids = pad_sequence(basic_token_type_ids, batch_first=True, padding_value=0)
        
        labels = torch.tensor([t for t in labels])
        
        return con_ids, isolate_ids, con_mask, isolate_mask, con_attention, isolate_attention, token_type_ids, basic_ids, basic_mask, basic_attention, basic_token_type_ids, labels

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


class MelbertProcessor(Processor):
    '''
        Process and Load data for ClassificationForBasicMean_Roberta Model
        args.model_name : name of model
        args.device : index of which device used
        args.ori_emb : True if contain unknown words
    '''
    def __init__(self, args, tokenizer, raw_data, type_='test'):
        super(MelbertProcessor, self).__init__(args, tokenizer, raw_data, type_)
        self.tokenized_input = []
        
        self._prepare_ids()
        self._sampler = self._return_sampler(type_)
        self._batch_sampler = BatchSampler(self._sampler(self.tokenized_input), self._batch_size, False)
        self._max_steps = len(self._batch_sampler)

    def _prepare_ids(self):
        logger.info('************ MelBERT Data Processor ***************')
        for sample in tqdm(self.raw_data):
            target = sample[0]
            sentence = sample[1].lower()
            index = sample[2]
            label = torch.tensor(int(sample[3]))
            pos = sample[4]

            # get tokens of context and target #
            con_ids_attetion = self.tokenizer(sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
            con_tokens = self.tokenizer.convert_ids_to_tokens(con_ids_attetion['input_ids'][0])
            con_attention = con_ids_attetion['attention_mask'][0]
            _, con_ni = tokenize_by_index(self.tokenizer, sentence, index)
            con_mask = torch.zeros(con_attention.shape)
            con_mask[con_ni[0]:con_ni[1]]=1
            token_type_ids = con_mask
            if not self.tokenizer.convert_tokens_to_ids(con_tokens)==_:
                print('*********** Token Split Error! ***********')
            
            # POS tag adding #
            pad_len=1
            if self.args.use_pos:
                pad_len = len(con_tokens)
                pos_token = self.tokenizer.tokenize(pos)
                con_tokens += pos_token + [self.tokenizer.sep_token]
                pad_len = len(con_tokens) - pad_len
                padding = (0, pad_len)
                con_attention = F.pad(con_attention, padding, value=1)
                con_mask = F.pad(con_mask, padding, value=0)
                token_type_ids = F.pad(token_type_ids, padding, value=3)     

            # set local content #
            if self.args.use_local_context:
                local_start = 1
                local_end = len(con_tokens) - pad_len
                comma1 = self.tokenizer.tokenize(",")[0]
                comma2 = self.tokenizer.tokenize(" ,")[0]
                for i, w in enumerate(con_tokens):
                    if i < con_ni[0] and (w in [comma1, comma2]):
                        local_start = i+1
                    if i > con_ni[1]-1 and (w in [comma1, comma2]):
                        local_end = i
                        break
                token_type_ids[local_start:local_end] = 2
                token_type_ids[con_ni[0]:con_ni[1]]=1
            else:
                token_type_ids[con_ni[0]:con_ni[1]]=1

            # get tokens of isolated target #
            isolate_ids_attetion = self.tokenizer(target,padding=True,truncation=True,max_length=512,return_tensors="pt")
            isolate_tokens = self.tokenizer.convert_ids_to_tokens(isolate_ids_attetion['input_ids'][0])
            isolate_attention = isolate_ids_attetion['attention_mask'][0]
            isolate_mask = torch.zeros(isolate_attention.shape)
            isolate_mask[1:-1]=1

            # convert tokens to ids #
            con_ids = self.tokenizer.convert_tokens_to_ids(con_tokens)
            con_ids = torch.tensor(con_ids, dtype=torch.long)
            isolate_ids = self.tokenizer.convert_tokens_to_ids(isolate_tokens)
            isolate_ids = torch.tensor(isolate_ids, dtype=torch.long)

            # change input type #
            token_type_ids = token_type_ids.int()
            
            #if (len(self.tokenized_input)) > 64:
                #break
            self.tokenized_input.append([con_ids, isolate_ids, con_mask, isolate_mask,
                                        con_attention, isolate_attention, token_type_ids, label])
        logger.info(f"=====Finished length: {len(self.tokenized_input)}!=====")

    def _collate_fn(self, batch):
        con_ids, isolate_ids, con_mask, isolate_mask, con_attention, isolate_attention, token_type_ids, labels = zip(*batch)

        con_ids = pad_sequence(con_ids, batch_first=True, padding_value=0)
        isolate_ids = pad_sequence(isolate_ids, batch_first=True, padding_value=0)
        con_mask = pad_sequence(con_mask, batch_first=True, padding_value=0)
        isolate_mask = pad_sequence(isolate_mask, batch_first=True, padding_value=0)
        con_attention = pad_sequence(con_attention, batch_first=True, padding_value=0)
        isolate_attention = pad_sequence(isolate_attention, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        labels = torch.tensor([t for t in labels])
        
        return con_ids, isolate_ids, con_mask, isolate_mask, con_attention, isolate_attention, token_type_ids, labels

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







        

    
