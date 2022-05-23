# -*- conding: utf-8 -*-
import os
from os.path import exists
import argparse
import torch
import time

class parse_args:
    def __init__(self):
        self.args = self.init_parse()
        self.set_cuda()
        
        self.set_timestamp() # set a timestamp
        
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
        parser.add_argument('--valset_dir', type=str, default='data/VUA18',
                            help='dir of val set.')
        parser.add_argument('--unk_emb', action='store_true',
                            help='input if contain unknown words.')
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
        parser.add_argument('--epochs', type=int, default=100,
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

        # Evaluating Configs
        parser.add_argument('--plot_dir', type=str, default='saves/plots',
                            help='dir of evaluation plots.')
        
        return parser.parse_args()

    def set_cuda(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.cuda_id
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.args.n_gpu = 1
        self.args.device = device

    def set_timestamp(self):
        now = int(round(time.time()*1000))
        now_str = time.strftime('%m-%d_%H:%M',time.localtime(now/1000)) # %Y-%m-%d %H:%M:%S '2017-11-07 16:47:14'
        print(now_str)
        self.args.stamp = now_str
        settings = vars(self.args)
        stamp_path = os.path.join('saves', self.args.stamp+'.txt')
        if not exists('saves'):
            os.makedirs(stamp_path)
        with open(stamp_path, 'w+') as f:
            for key in settings.keys():
                f.write(key)
                f.write(':  ')
                f.write(str(settings[key]))
                f.write('\n')
                




