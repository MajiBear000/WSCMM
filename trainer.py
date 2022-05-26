# -*- conding: utf-8 -*-
import os
from os.path import exists
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import trange
from utils import compute_metrics, output_param, log_results, loss_plot, acc_plot
from torch.nn.utils.rnn import pad_sequence

def get_input_from_batch(args, batch):
    if args.model_name == 'roberta':
        inputs = {  'basic_ids': batch[0],
                    'basic_attention': batch[1],
                    'basic_mask': batch[2],
                    'con_ids': batch[3],
                    'con_attention': batch[4],
                    'con_mask': batch[5],
                    }
        labels = batch[6]
    elif args.model_name == 'linear':
        inputs = {  'basic_emb': batch[0],
                    'test_emb': batch[1],
            }
        labels = batch[2]
    return inputs, labels

class Trainer(object):
    """ Trainer """
    def __init__(self, args, train_data, val_data, model):
        self.args = args
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.train_loss = []
        self.val_loss = []
        self.results = []
        self.f1 = []
        self.acc = []
        self.pre = []
        self.rec = []
        
        self.setup_train()

    def setup_train(self):
        """ set up a trainer """
        train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.args.train_batch_size, collate_fn=self.collate_fn)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        #self.optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        self.device = self.args.device
        self.model.to(self.device)
        self.model.double()
        self.loss_func = F.cross_entropy
    
    def train(self):
        # Train
        print("******************** Running training **********************")
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            tr_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs, labels = get_input_from_batch(self.args, batch)
                logits  = self.model(**inputs)
            
                loss = self.loss_func(logits, labels)
                tr_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()

                if (step+1) % 1000 == 0:
                    print(f"Train loss for {int((step+1)/1000)}000 step: {tr_loss}")
            self.train_loss.append(tr_loss/(step+1))

            self.evaluate(self.val_data, val=True)
        if not exists(self.args.plot_dir):
            os.makedirs(self.args.plot_dir)
        self.log_plot()
        #loss_plot(self.args, self.train_loss, self.val_loss)
        #acc_plot(self.args, self.pre, self.rec, self.f1, self.acc)
        #output_param(model)
        return self.train_loss, self.val_loss, self.results


    def evaluate(self, test_data, val=False):
        """ set up evaluater """
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.test_batch_size, collate_fn=self.collate_fn)
        
        preds = None
        t_loss = 0
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = get_input_from_batch(self.args, batch)
            with torch.no_grad():
                logits = self.model(**inputs)
                loss = self.loss_func(logits, labels)
                t_loss += loss.item()

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        log_results(result, val)
        if val:
            self.log_metrix(result)
            self.results.append(result)
            self.val_loss.append(t_loss/(step+1))
        return result

    def log_metrix(self, result):
        self.f1.append(result['f1'])
        self.acc.append(result['acc'])
        self.pre.append(result['precision'])
        self.rec.append(result['recall'])

    def log_plot(self):
        plt.figure(figsize=(14,6))

        plt.subplot(121)
        plt.plot(self.train_loss, label='Train loss')
        plt.plot(self.val_loss, label='Dev loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(122)
        plt.plot(self.pre, label='Precision')
        plt.plot(self.rec, label='Recall')
        plt.plot(self.f1, label='F1-score')
        plt.plot(self.acc, label='Accuracy')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.legend()
        
        self.plot_path = os.path.join(self.args.plot_dir, self.args.stamp+'_plot.png')
        plt.savefig(self.plot_path)

    def collate_fn(self, batch):
        basic_ids,basic_attention,basic_mask,con_ids,con_attention,con_mask,labels = zip(*batch)

        basic_ids = pad_sequence(basic_ids, batch_first=True, padding_value=0)
        basic_attention = pad_sequence(basic_attention, batch_first=True, padding_value=0)
        basic_mask = pad_sequence(basic_mask, batch_first=True, padding_value=0)
        con_ids = pad_sequence(con_ids, batch_first=True, padding_value=0)
        con_attention = pad_sequence(con_attention, batch_first=True, padding_value=0)
        con_mask = pad_sequence(con_mask, batch_first=True, padding_value=0)
        labels = torch.tensor([t for t in labels])

        return basic_ids,basic_attention,basic_mask,con_ids,con_attention,con_mask,labels









