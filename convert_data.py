# read labeled data and convert to unlabeled data
# e.g. from "sentence 1   sentence 2  label" to "sentence1\n sentence2\n"

import os

tasks = ["Abt-Buy", "DBLP-ACM", "DBLP-GoogleScholar", "Walmart-Amazon", "Amazon-Google", "DBLP-ACM-dirty", "DBLP-GoogleScholar-dirty", "Walmart-Amazon-dirty"]

for task in tasks:
    path = 'data/em/%s' % task
    print(path)
    train_path = os.path.join(path, 'train.txt')
    nolabel_train_path = os.path.join(path, 'train_no_label.txt')
    output_file = open(nolabel_train_path, 'w')
    for line in open(train_path):
        s1, s2, label = line.strip().split('\t')
        output_file.write(s1+"\n")
        output_file.write(s2+"\n")
    output_file.close()
    print(f"new file %s written", nolabel_train_path)
