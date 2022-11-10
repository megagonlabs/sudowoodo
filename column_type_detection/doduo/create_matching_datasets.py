import os
import sys
import csv
import random
import numpy as np
import pickle

from collections import Counter
csv.field_size_limit(sys.maxsize)

if __name__ == '__main__':
    pairs = pickle.load(open('blocking_result.pkl', 'rb'))

    columns = open('columns_labeled.txt').read().split('\n')

    labels = {}
    max_len = 64
    for i in range(5):
        reader = csv.DictReader(open('sato_cv_%d.csv' % i))
        for column in reader:
            tokens = column['data'].split(' ')
            data = ' '.join(tokens[:max_len])
            labels[data] = column['class']

    all_examples = []
    all_labels = []
    for left_id, right_id, score in pairs:
        if left_id == right_id:
            continue

        left, right = columns[left_id], columns[right_id]
        label = int(labels[left] == labels[right])
        all_labels.append(label)
        all_examples.append('%s\t%s\t%d\n' % (left, right, label))

    random.shuffle(all_examples)
    fns = ['train.txt', 'valid.txt', 'test.txt']
    starts = [0, 1000, 1500]
    ends = [1000, 1500, 2000]

    for fn, start, end in zip(fns, starts, ends):
        with open(fn, 'w') as fout:
            for line in all_examples[start:end]:
                fout.write(line)

    with open('all_pairs.txt', 'w') as fout:
        for line in all_examples:
            fout.write(line)

    print('num_pairs =', len(all_examples))
    print('pos_rate =', sum(all_labels) / len(all_labels))
