# -*- conding: utf-8 -*-
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import trange
from utils import compute_metrics, output_param

def get_input_from_batch(args, batch):
    inputs = {  'basic_emb': batch[0],
                'test_emb': batch[1],
        }
    labels = batch[2]
    return inputs, labels

def train(args, train_data, test_data, model):
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    device = args.device
    model.to(device)
    model.double()
    
    # Train
    print("******************** Running training **********************")
    for epoch in trange(int(args.epochs), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            logit  = model(**inputs)
            
            loss = F.cross_entropy(logit, labels)
            tr_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            if (step+1) % 1000 == 0:
                print(f"Train loss for {int((step+1)/1000)}000 step: {tr_loss}")
                tr_loss = 0

        evaluate(args, test_data, model)
    #output_param(model)
    return 0


def evaluate(args, test_data, model):
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    preds = None
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = get_input_from_batch(args, batch)
        with torch.no_grad():
            logits = model(**inputs)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)

    














