# -*- conding: utf-8 -*-
import os
import logging
import tqdm
import sys
sys.path.append("..")

from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW

from tqdm import trange, tqdm
from sw.utils import compute_metrics, output_param, log_results

logger = logging.getLogger(__name__)


def get_input_from_batch(model_name, batch):
    if model_name == 'roberta':
        inputs = {  'basic_ids': batch[0],
                    'basic_attention': batch[1],
                    'basic_mask': batch[2],
                    'con_ids': batch[3],
                    'con_attention': batch[4],
                    'con_mask': batch[5],
                    }
        labels = batch[6]
    elif model_name == 'melbert':
        inputs = {  'input_ids': batch[0],
                    'input_ids_2': batch[1],
                    'target_mask': batch[2],
                    'target_mask_2': batch[3],
                    'attention_mask': batch[4],
                    'attention_mask_2': batch[5],
                    'token_type_ids': batch[6],
                    }
        labels = batch[7]
    elif model_name == 'linear':
        inputs = {  'basic_emb': batch[0],
                    'test_emb': batch[1],
            }
        labels = batch[2]
    return inputs, labels

class Trainer(object):
    """
        Trainer
            args.model_name : name of the model used to train.
            args.device : the device where model trained on.
            args.lr : learning rate.
            args.adam_epsilon : Epsilon for Adam optimizer.
            args.max_grad_norm : Max gradient norm.
            args.plot_dir : evaluation plot save dir.
            args.stamp : timestamp used to identify saved results.

            args.save_path : path to where experiments saved
            
    """
    def __init__(self, args, data, model):
        self.args = args
        self.train_data = data['train']
        self.test_data = data['test']
        self.val_data = data['val']
        self.model = model
        self._train_loss = []
        self._val_loss = []
        self._results = []
        self._f1 = []
        self._acc = []
        self._pre = []
        self._rec = []
        
        self._setup_train()

    def _setup_train(self):
        """ set up a trainer """
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr, momentum=0.9)
        #self.optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        self.device = self.args.device
        self.model.to(self.device)
        self.model.double()
        self.loss_func = F.cross_entropy
    
    def train(self):
        # Train
        logger.info("***** Running training *****")
        for epoch in range(int(self.args.epochs)):
            logger.info(f"===== Epoches: {epoch+1} =====")
            tr_loss = 0
            for step, batch in tqdm(enumerate(self.train_data), total=len(self.train_data), desc="Epoch"):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs, labels = get_input_from_batch(self.args.model_name,
                                                      batch)
                logits  = self.model(**inputs)
            
                loss = self.loss_func(logits, labels).to(self.args.device)
                if self.args.n_gpu>1:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                tr_loss += loss.item()

                if (step+1) % 1000 == 0:
                    logger.info(f"Train loss for {int((step+1)/1000)}000 step: {tr_loss}")
            self._train_loss.append(tr_loss/(step+1))
            if self.val_data is None:
                self.evaluate(val=True)
            else:
                self.evaluate(test_data=self.val_data, val=True)
        if not exists(self.args.plot_dir):
            os.makedirs(self.args.plot_dir)
        self._log_plot()
        #output_param(model)
        return self._train_loss, self._val_loss, self._results


    def evaluate(self, test_data=None, val=False):
        if test_data is None:
            test_data = self.test_data
        
        preds = None
        t_loss = 0
        for step, batch in enumerate(test_data):
            #batch = tuple(t.to(self.device) for t in batch)
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs, labels = get_input_from_batch(self.args.model_name, batch)
            with torch.no_grad():
                logits = self.model(**inputs)
                loss = self.loss_func(logits, labels)
                t_loss += loss.item()

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(),
                                      axis=0)
                    out_label_ids = np.append(out_label_ids,
                                              labels.detach().cpu().numpy(),
                                              axis=0)
        
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        log_results(result, val)
        if val:
            self._log_metrix(result)
            self._results.append(result)
            self._val_loss.append(t_loss/(step+1))
        return result

    def _log_metrix(self, result):
        self._f1.append(result['f1'])
        self._acc.append(result['acc'])
        self._pre.append(result['precision'])
        self._rec.append(result['recall'])

    def _log_plot(self):
        '''log evaluation plot to plot save dir'''
        plt.figure(figsize=(14,6))

        plt.subplot(121)
        plt.plot(self._train_loss, label='Train loss')
        plt.plot(self._val_loss, label='Dev loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(122)
        plt.plot(self._pre, label='Precision')
        plt.plot(self._rec, label='Recall')
        plt.plot(self._f1, label='F1-score')
        plt.plot(self._acc, label='Accuracy')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.legend()
        
        self._plot_path = os.path.join(self.args.plot_dir,
                                      self.args.stamp+'_plot.png')
        plt.savefig(self._plot_path)









