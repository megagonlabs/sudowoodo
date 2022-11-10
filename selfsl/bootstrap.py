import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils import data
from .dataset import DMDataset
from .bt_dataset import BTDataset
from .block import run_blocking
from sklearn.metrics import confusion_matrix

def bootstrap(model, hp, blocked=True):
    """Construct a new training set based on similarity for em
    """
    pos_factor = neg_factor = hp.multiplier

    train_path = os.path.join('data/%s/%s/train.txt' % \
                              (hp.task_type, hp.task))
    # blocking results are available
    dataset = DMDataset(train_path,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=None,
                         da=None)

    if blocked:
        train_iter = data.DataLoader(dataset=dataset,
                               batch_size=hp.batch_size//2,    # half of barlow twins'
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DMDataset.pad)

        all_sims = []
        for i, batch in enumerate(train_iter):
            x1, x2, _, _ = batch

            # compute similarity
            x1 = x1.to(model.device)
            x2 = x2.to(model.device)
            with torch.no_grad():
                batch_size = len(x1)
                x = torch.cat((x1, x2), dim=0)
                z = model.projector(model.bert(x)[0][:, 0, :])
                z = F.normalize(z, dim=1)
                z1 = z[:batch_size]
                z2 = z[batch_size:]
                similarity = torch.sum(z1 * z2, dim=-1)
                all_sims += similarity.cpu().numpy().tolist()
    else:
        # blocking result is not available: construct dataset live
        path = 'data/em/%s' % hp.task

        left_path = os.path.join(path, 'tableA.txt')
        right_path = os.path.join(path, 'tableB.txt')

        # BT blocking
        left_dataset = BTDataset(left_path,
                                 lm=hp.lm,
                                 size=None,
                                 da='del',
                                 max_len=hp.max_len)


        right_dataset = BTDataset(right_path,
                                  lm=hp.lm,
                                  size=None,
                                  da='del',
                                  max_len=hp.max_len)

        pairs = run_blocking(left_dataset, right_dataset, model, hp)

        # reset dataset with new blocking results
        all_sims = []
        dataset.pairs = []
        dataset.labels = []
        for lid, rid, score in pairs:
            all_sims.append(score)
            dataset.pairs.append((left_dataset.instances[lid],
                                  right_dataset.instances[rid]))

        threshold = sorted(all_sims)[int(len(all_sims) * (1.0-1.0/hp.k))]
        for sim in all_sims:
            if sim < threshold:
                dataset.labels.append(0)
            else:
                dataset.labels.append(1)


    # all_jaccard = []
    # for s1, s2 in dataset.pairs:
    #     t1 = set(s1.split())
    #     t2 = set(s2.split())
    #     jaccard = len(t1 & t2) / len(t1 | t2)
    #     all_jaccard.append(jaccard)
    # all_sims = all_jaccard

    # get similarity threshold
    N = len(dataset)
    pos_threshold = 0.0
    neg_threshold = 1.0
    pos_sims, neg_sims = [], []

    # method 1
    # for i in range(min(hp.size, N)):
    #     if dataset.labels[i] == 0:
    #         neg_sims.append(all_sims[i])
    #         pos_threshold = max(pos_threshold, all_sims[i])
    #     else:
    #         pos_sims.append(all_sims[i])
    #         neg_threshold = min(neg_threshold, all_sims[i])

    # method 2
    # pos_threshold = np.mean(pos_sims)
    # neg_threshold = np.mean(neg_sims)

    # method 3
    pos_cnt = 0
    neg_cnt = 0
    for i in range(min(hp.size, N)):
        if dataset.labels[i] == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    new_pos_cnt = pos_cnt * (pos_factor - 1)
    new_neg_cnt = neg_cnt * (neg_factor - 1)
    sims_sorted = sorted(all_sims[hp.size:])
    pos_threshold = sims_sorted[-new_pos_cnt]
    neg_threshold = sims_sorted[new_neg_cnt]

    # label all
    new_pairs, new_labels = [], []
    ground_truth = []
    correct = 0
    for i in range(N):
        if i < hp.size and not hp.zero:
            new_pairs.append(dataset.pairs[i])
            new_labels.append(dataset.labels[i])
            ground_truth.append(dataset.labels[i])
            correct += 1
        else:
            if all_sims[i] > pos_threshold:
                new_pairs.append(dataset.pairs[i])
                new_labels.append(1)
                ground_truth.append(dataset.labels[i])
                if dataset.labels[i] == 1:
                    correct += 1

            if all_sims[i] < neg_threshold:
                new_pairs.append(dataset.pairs[i])
                new_labels.append(0)
                ground_truth.append(dataset.labels[i])
                if dataset.labels[i] == 0:
                    correct += 1

    # debug
    print(pos_threshold, neg_threshold, pos_threshold > neg_threshold, len(new_pairs))

    print('original_ratio =', sum(dataset.labels[:hp.size]) / hp.size)
    print('new_ratio =', sum(new_labels) / len(new_labels))
    print('acc =', correct / len(new_labels))
    # print('(tn, fp, fn, tp) =', confusion_matrix(ground_truth, new_labels).ravel())
    TN, FP, FN, TP = confusion_matrix(ground_truth, new_labels).ravel()
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    print('(tpr, tnr, fpr, fnr) =', (TPR, TNR, FPR, FNR))

    dataset.pairs = new_pairs
    dataset.labels = new_labels
    return dataset, TPR, TNR, FPR, FNR


def bootstrap_cleaning(model, hp, pos_factor=4, neg_factor=4):
    """Construct a new training set based on similarity for data cleaning
    """
    train_path = os.path.join('data/%s/%s/train.txt' % \
                              (hp.task_type, hp.task))

    dataset = DMDataset(train_path,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         da=None)

    # add unlabeled data to the dataset
    train_no_label_path = os.path.join('data/%s/%s/train_no_label.txt' % \
                              (hp.task_type, hp.task))

    for line in open(train_no_label_path):
        dataset.pairs.append(line.strip())
        dataset.labels.append(-1) # undecided


    train_iter = data.DataLoader(dataset=dataset,
                           batch_size=hp.size, # the first batch will be the labeled
                           shuffle=False,
                           num_workers=0,
                           collate_fn=DMDataset.pad)

    all_embeds = None
    all_sims = []
    all_labels = []
    all_indices = []

    for i, batch in enumerate(train_iter):
        x, _ = batch

        # compute similarity
        x = x.to(model.device)
        with torch.no_grad():
            batch_size = len(x)
            z = model.projector(model.bert(x)[0][:, 0, :])
            z = F.normalize(z, dim=1)
            if all_embeds is None:
                all_embeds = z
            else:
                # compute similarity
                sim_mat = z.matmul(all_embeds.transpose(0, 1))
                sims, indices = sim_mat.max(dim=1)
                # print(sims, indices)
                all_sims += sims.cpu().numpy().tolist()
                all_indices += indices.cpu().numpy().tolist()
                for idx in indices:
                    all_labels.append(dataset.labels[idx])

    # balanced
    N = len(dataset.pairs)
    pos_cnt = 0
    neg_cnt = 0
    for i in range(hp.size):
        if dataset.labels[i] == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    new_pos_cnt = pos_cnt * (pos_factor - 1)
    new_neg_cnt = neg_cnt * (neg_factor - 1)

    pos_sims = []
    neg_sims = []

    for i in range(hp.size, N):
        idx = i - hp.size
        if all_labels[idx] == 0:
            neg_sims.append(all_sims[idx])
        else:
            pos_sims.append(all_sims[idx])

    neg_sims.sort(reverse=True)
    neg_threshold = neg_sims[min(new_neg_cnt, len(neg_sims) - 1)]

    pos_sims.sort(reverse=True)
    pos_threshold = pos_sims[min(new_pos_cnt, len(pos_sims) - 1)]

    # label all
    new_pairs, new_labels = [], []
    for i in range(N):
        if i < hp.size:
            new_pairs.append(dataset.pairs[i])
            new_labels.append(dataset.labels[i])
        else:
            idx = i - hp.size
            if all_labels[idx] == 0 and all_sims[idx] >= neg_threshold:
                new_pairs.append((dataset.pairs[i],))
                new_labels.append(0)

            if all_labels[idx] == 1 and all_sims[idx] >= pos_threshold:
                new_pairs.append((dataset.pairs[i],))
                new_labels.append(1)

    # debug
    print('original_ratio =', sum(dataset.labels[:hp.size]) / hp.size)
    print('new_ratio =', sum(new_labels) / len(new_labels))

    dataset.pairs = new_pairs
    dataset.labels = new_labels
    return dataset
