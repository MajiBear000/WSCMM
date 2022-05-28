# -*- conding: utf-8 -*-

class RawData(object):
    def __init__(self, name):
        self.name=name

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
           
