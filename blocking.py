import os
import argparse
import json
import sys
import numpy as np
import random
import torch
import csv

from selfsl.bt_dataset import BTDataset
from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR
from selfsl.block import *
from torch.utils import data
from apex import amp
from sklearn.metrics import recall_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def load_model(hp):
    """Load the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp,
                        device=device,
                        lm=hp.lm)
    if not hp.lm_only:
        saved_state = torch.load(hp.ckpt_path,
                    map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state['model'])

    model = model.to(device)
    if hp.fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return model


def tfidf_blocking(pathA, pathB, K=10):
    # read csv
    tableA = []
    tableB = []

    reader = csv.DictReader(open(pathA))
    for row in reader:
        tableA.append(' '.join(row.values()))

    reader = csv.DictReader(open(pathB))
    for row in reader:
        tableB.append(' '.join(row.values()))

    corpus = tableA + tableB
    vectorizer = TfidfVectorizer().fit(corpus)

    matA = vectorizer.transform(tableA).toarray()
    matB = vectorizer.transform(tableB).toarray()

    res = blocked_matmul(matA, matB, k=K)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DBLP-ACM")
    parser.add_argument("--task_type", type=str, default="em")
    parser.add_argument("--logdir", type=str, default="results/")
    parser.add_argument("--ckpt_path", type=str, default=None) # em_Abt-Buy_da=barlow_twins_id=0_size=300_ssl.pt
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--projector", type=str, default='768')
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--tfidf", dest="tfidf", action="store_true") # if set, apply the baseline blocker
    parser.add_argument("--lm_only", dest="lm_only", action="store_true") # if set, only apply a non-fine-tuned LM
    hp = parser.parse_args()

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    path = 'data/em/%s' % hp.task

    left_path = os.path.join(path, 'tableA.txt')
    right_path = os.path.join(path, 'tableB.txt')

    if hp.tfidf:
        # tfidf blocking
        pairs = tfidf_blocking(left_path.replace('.txt', '.csv'),
                               right_path.replace('.txt', '.csv'), K=hp.k)
    else:
        # BT blocking
        left_dataset = BTDataset(left_path,
                                 lm=hp.lm,
                                 size=None,
                                 max_len=hp.max_len)

        right_dataset = BTDataset(right_path,
                                  lm=hp.lm,
                                  size=None,
                                  max_len=hp.max_len)

        model = load_model(hp)
        pairs = run_blocking(left_dataset, right_dataset, model, hp)

    # dump pairs
    import pickle
    pickle.dump(pairs, open('blocking_result.pkl', 'wb'))

    # ground_truth, total = read_ground_truth(path)
    # if hp.k:
    #     for k in range(1, hp.k+1):
    #         recall, size = evaluate_pairs(pairs, ground_truth, k=k)
    #         print('k=%d,recall=%f,size=%d' % (k, recall, size))
    # else:
    #     recall = evaluate_pairs(pairs, ground_truth)
    #     print('recall = %f, original_size = %d, new_size = %d' % (recall, total, len(pairs)))
