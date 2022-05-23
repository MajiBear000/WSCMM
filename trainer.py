# -*- conding: utf-8 -*-
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import trange
from utils import compute_metrics, output_param, log_results

def get_input_from_batch(batch):
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
        
        self.setup_train()

    def setup_train(self):
        """ set up a trainer """
        train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)
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
                inputs, labels = get_input_from_batch(batch)
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
        #output_param(model)
        return self.train_loss, self.val_loss, self.results


    def evaluate(self, test_data, val=False):
        """ set up evaluater """
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.test_batch_size)
        
        preds = None
        t_loss = 0
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            inputs, labels = get_input_from_batch(batch)
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
            self.results.append(result)
            self.val_loss.append(t_loss/(step+1))

    














