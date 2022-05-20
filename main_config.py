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
        
        return parser.parse_args()
