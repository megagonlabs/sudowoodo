import os
import sys
import csv
import random
import numpy as np

from collections import Counter
csv.field_size_limit(sys.maxsize)

if __name__ == '__main__':
    os.system('wget https://doduo-data.s3-us-west-2.amazonaws.com/data.tar.gz')
    os.system('tar -zvxf data.tar.gz')
    columns = []
    for i in range(5):
        reader = csv.DictReader(open('data/sato_cv_%d.csv' % i))
        for column in reader:
            columns.append(column)

    max_len = 64
    used = set([])
    with open('columns_labeled.txt', 'w') as fout_labeled, \
         open('columns.txt', 'w') as fout:
        for col in columns:
            tokens = col['data'].split(' ')
            data = ' '.join(tokens[:max_len])
            if data not in used:
                used.add(data)
                fout_labeled.write('%s %s %s\n' % (data, col['class'], col['class_id']))
                fout.write('%s\n' % data)

    print(len(used))
