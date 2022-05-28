import csv
import os
import re

from sw.data import RawData

def load_data(args):
    dataset = RawData(args.dataset_name)
    train_path = os.path.join(args.trainset_dir, 'train.tsv')
    test_path = os.path.join(args.testset_dir, 'test.tsv')
    val_path = os.path.join(args.valset_dir, 'val.tsv')
    if args.dataset_name == 'vua18' or 'vua20':
        dataset['train'] = _read_vua(train_path, re.search('VUA18', train_path))
        dataset['test'] = _read_vua(test_path, re.search('VUA18', test_path))
        dataset['val'] = _read_vua(val_path, re.search('VUA18', val_path))
    return dataset

def _read_vua(file_path, no_fgpos=None):
    dataset = []
    with open(file_path, encoding='utf8') as f:
        lines = csv.reader(f, delimiter='\t')
        next(lines)
        w_index = 0
        flag = True
        for line in lines:
            sen_id = line[0]
            sentence = line[2]

            if no_fgpos:
                ind = line[4]
            else:
                ind = line[5]
            label = line[1]

            index = int(ind)
            word = sentence.split()[index]

            dataset.append([word, sentence, index, label])
    print(file_path, len(dataset))
    return dataset

