import os
import numpy as np
import random
import torch
import csv

from .bt_dataset import BTDataset
from .utils import blocked_matmul

from torch.utils import data
from sklearn.metrics import recall_score
from tqdm import tqdm


def encode_all(dataset, model, hp):
    """Encode all records using to the model.
    """
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=dataset.pad)

    all_encs = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            x, _ = batch
            x = x.to(model.device)
            if hasattr(hp, "lm_only") and hp.lm_only:
                enc = model.bert(x)[0][:, 0, :]
            else:
                enc = model.bert(x)[0][:, 0, :]
                # enc = model.projector(enc)
            all_encs += enc.cpu().numpy().tolist()

    res = np.array(all_encs)
    res = [v / np.linalg.norm(v) for v in res]
    return res




def run_blocking(left_dataset, right_dataset, model, hp):
    """Run the Barlow Twins blocking method.

    Args:
        left_dataset (BTDataset): the left table
        right_dataset (BTDataset): the right table
        model (BarlowTwinsSimCLR): the BT/SimCLR model
        hp (Namespace): hyper-parameters

    Returns:
        list of tuple: the list of candidate pairs
    """
    # encode both datasets
    mata = encode_all(left_dataset, model, hp)
    matb = encode_all(right_dataset, model, hp)

    # matmul to compute similarity
    pairs = blocked_matmul(mata, matb,
                           threshold=hp.threshold,
                           k=hp.k,
                           batch_size=hp.batch_size)
    return pairs


def read_ground_truth(path):
    """Read groundtruth matches from train/valid/test sets.

    Args:
        path (str): the path to the datasets

    Returns:
        List of tuple: matched pairs
        int: the total number of original match / non-match
    """
    res = []
    total = 0
    for fn in ['train.csv', 'valid.csv', 'test.csv']:
        reader = csv.DictReader(open(os.path.join(path, fn)))
        for row in reader:
            lid = int(row['ltable_id'])
            rid = int(row['rtable_id'])
            lbl = row['label']
            if int(lbl) == 1:
                res.append((lid, rid))
            total += 1
    return res, total

def evaluate_pairs(pairs, ground_truth, k=None):
    """Return the recall given the set of pairs and ground truths.

    Args:
        pairs (list): the computed list
        ground_truth (list): the ground truth list
        k (int, optional): if set, compute recall only for
                           the top k for each right index

    Returns:
        float: the recall
    """
    if k:
        r_index = {}
        for l, r, score in pairs:
            if r not in r_index:
                r_index[r] = []
            r_index[r].append((score, l))

        pairs = []
        for r in r_index:
            r_index[r].sort(reverse=True)
            for _, l in r_index[r][:k]:
                pairs.append((l, r))

        pairs = set(pairs)
        y_true = [1 for _ in ground_truth]
        y_pred = [int(p in pairs) for p in ground_truth]
        return recall_score(y_true, y_pred), len(pairs)
    else:
        print('pairs =', len(pairs), pairs[:10])
        print('ground_truth =', len(ground_truth))
        pairs = [(l, r) for l, r, _ in pairs]
        pairs = set(pairs)
        y_true = [1 for _ in ground_truth]
        y_pred = [int(p in pairs) for p in ground_truth]
        return recall_score(y_true, y_pred)


def evaluate_blocking(model, hp):
    """Evaluate an embedding model for blocking.

    Args:
        model (BarlowTwinsSimCLR): the embedding model
        hp (NameSpace): hyper-parameters

    Returns:
        List of float: the list of recalls
        List of int: the list of candidate set sizes
    """
    path = 'data/em/%s' % hp.task

    left_path = os.path.join(path, 'tableA.txt')
    right_path = os.path.join(path, 'tableB.txt')

    # BT blocking
    print('encode left')
    left_dataset = BTDataset(left_path,
                             lm=hp.lm,
                             size=None,
                             max_len=hp.max_len)

    print('encode right')
    right_dataset = BTDataset(right_path,
                              lm=hp.lm,
                              size=None,
                              max_len=hp.max_len)

    print('blocked MM:')
    pairs = run_blocking(left_dataset, right_dataset, model, hp)

    print('Read ground truth')
    ground_truth, total = read_ground_truth(path)

    if hp.k:
        recalls, sizes = [], []
        for k in tqdm(range(1, hp.k+1)):
            recall, size = evaluate_pairs(pairs, ground_truth, k)
            recalls.append(recall)
            sizes.append(size)
        return recalls, sizes
    else:
        recall = evaluate_pairs(pairs, ground_truth)
        return recall, len(pairs)
