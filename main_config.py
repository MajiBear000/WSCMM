# -*- conding: utf-8 -*-
import argparse

class parse_args:
    def __init__(self):
        self.args = self.init_parse()
        
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
        
        return parser.parse_args()
