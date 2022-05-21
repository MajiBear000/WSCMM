import csv
import os

def load_data(args):
    dataset = {'train':None,'test':None,'val':None}
    train_path = os.path.join(args.trainset_dir, 'train.tsv')
    test_path = os.path.join(args.testset_dir, 'test.tsv')
    if args.dataset_name == 'vua18' or 'vua20':
        dataset['train'] = read_vua(train_path)
        dataset['test'] = read_vua(test_path)
    return dataset

def read_vua(file_path, no_fgpos=False):
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
