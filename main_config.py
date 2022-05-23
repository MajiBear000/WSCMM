# -*- conding: utf-8 -*-
import os
import argparse
import torch

class parse_args:
    def __init__(self):
        self.args = self.init_parse()
        self.set_cuda()
        
    def init_parse(self):
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument('--dataset_name', type=str, default='vua20',
                            choices=['vua18','vua20', 'trofi', 'mohx'],
                            help='name of dataset.')
        parser.add_argument('--trainset_dir', type=str, default='data/VUA20',
                            help='dir of training set.')
        parser.add_argument('--testset_dir', type=str, default='data/VUA20',
                            help='dir of test set.')
        parser.add_argument('--model_path', type=str, default='senseCL/checkpoint/checkpoint-1200',
                            help='path for pretrained model.')
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed for initialization.')
        parser.add_argument('--cuda_id', type=str, default='1',
                            help='Which cuda use to training.')
        parser.add_argument('--no_cuda', action='store_true',
                            help='input if dont use cuda.')


        # Training and Model Configs
        parser.add_argument('--drop_ratio', type=float, default=0.2,
                            help='ratio of dropout layer.')
        parser.add_argument('--bias', type=float, default=0.2,
                            help='bias of meodel.')
        parser.add_argument('--num_train_epochs', type=int, default=3,
                            help='number of set training epochs.')
        parser.add_argument('--train_batch_size', type=int, default=32,
                            help='size of training batch.')
        parser.add_argument('--test_batch_size', type=int, default=64,
                            help='size of testing batch.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='The initial learning rate for Adam.')
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        
        return parser.parse_args()

    def set_cuda(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.cuda_id
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.args.n_gpu = 1
        self.args.device = device

        
