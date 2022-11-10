from pathlib import Path
import copy
import argparse
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import mlflow

from torch import nn, optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils import data
from apex import amp
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .augment import Augmenter
from .bt_dataset import BTDataset
from .dataset import DMDataset
from .block import evaluate_blocking
from .bootstrap import bootstrap, bootstrap_cleaning

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsSimCLR(nn.Module):
    # the encoder is bert+projector
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-')))
        self.projector = nn.Linear(hidden_size, sizes[-1])

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        # a fully connected layer for fine tuning
        #self.fc = torch.nn.Linear(hidden_size * 2, 2)
        if hp.task_type == 'em':
            self.fc = nn.Linear(sizes[-1] * 2, 2)
        else:
            self.fc = nn.Linear(sizes[-1], 2)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels


    def forward(self, flag, y1, y2, y12, da=None, cutoff_ratio=0.1):
        if flag in [0, 1]:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            if da == 'cutoff':
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)

                # modify the word embeddings of y2
                # l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                # s = random.randint(0, seq_len - l - 1)

                # y2_word_embeds[:, s:s+l, :] = 0.0

                # modify the position embeddings of y2
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                # position_ids = self.bert.embeddings.position_ids[:, :seq_len]
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                # sample again
                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                # merge y1 and y2
                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                # cat y1 and y2 for faster training
                y = torch.cat((y1, y2))
                z = self.bert(y)[0][:, 0, :]
            z = self.projector(z)

            if flag == 0:
                # simclr
                logits, labels = self.info_nce_loss(z, batch_size, 2)
                loss = self.criterion(logits, labels)
                return loss
            elif flag == 1:
                # barlow twins
                z1 = z[:batch_size]
                z2 = z[batch_size:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # sum the cross-correlation matrix between all gpus
                #c.div_(self.hp.batch_size)
                #torch.distributed.all_reduce(c)

                # use --scale-loss to multiply the loss by a constant factor
                # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.hp.scale_loss)
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                # off_diag = off_diagonal(c).pow_(2).sum().mul(self.hp.scale_loss)
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
        elif flag == 2:
            # fine tuning
            if self.hp.task_type == 'em':
                x1 = y1
                x2 = y2
                x12 = y12

                x1 = x1.to(self.device) # (batch_size, seq_len)
                x2 = x2.to(self.device) # (batch_size, seq_len)
                x12 = x12.to(self.device) # (batch_size, seq_len)
                # left+right
                enc_pair = self.projector(self.bert(x12)[0][:, 0, :]) # (batch_size, emb_size)
                #enc_pair = self.bert(x12)[0][:, 0, :] # (batch_size, emb_size)
                batch_size = len(x1)
                # left and right
                enc = self.projector(self.bert(torch.cat((x1, x2)))[0][:, 0, :])
                #enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
                enc1 = enc[:batch_size] # (batch_size, emb_size)
                enc2 = enc[batch_size:] # (batch_size, emb_size)
                # return self.fc(torch.cat((enc1, enc2, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
                return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
            else: # cleaning
                x1 = y1
                x1 = x1.to(self.device) # (batch_size, seq_len)
                enc = self.projector(self.bert(x1)[0][:, 0, :]) # (batch_size, emb_size)
                return self.fc(enc)

def evaluate(model, iterator, threshold=None, ec_task=None, dump=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class
        ec_task (string, optional): if set, evaluate error correction
        dump (boolean, optional): if true, dump the test results

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            if len(batch) == 4:
                x1, x2, x12, y = batch
                logits = model(2, x1, x2, x12)
            else:
                x, y = batch
                logits = model(2, x, None, None)

            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        if dump:
            import pickle
            pickle.dump(pred, open('test_results.pkl', 'wb'))
            mlflow.log_artifact('test_results.pkl')

        f1 = metrics.f1_score(all_y, pred, zero_division=0)
        p = metrics.precision_score(all_y, pred, zero_division=0)
        r = metrics.recall_score(all_y, pred, zero_division=0)
        # error correction
        if ec_task:
            path = 'data/cleaning/%s/test.txt.ec' % ec_task
            # new_pred = pred.copy()
            current = 0
            indices = open(path).readlines()
            indices = [idx[:-1].split('\t') for idx in indices]
            tp, fp, fn = 0.0, 0.0, 0.0

            while current < len(pred):
                start = current
                while current + 1 < len(pred) and \
                      indices[current][:2] == indices[current + 1][:2]:
                    current += 1
                
                max_idx = -1
                max_prob = 0.0
                for idx in range(start, current+1):
                    if all_probs[idx] > max_prob:
                        max_prob = all_probs[idx]
                        max_idx = idx
                    
                    # new_pred[idx] = 0

                # predict
                original, res, ground_truth = indices[max_idx][2:]
                correction = res if pred[max_idx] == 1 else original

                # if start == 0:
                #     print(original, res, ground_truth)

                if original != ground_truth and correction == ground_truth:
                    tp += 1
                if original == ground_truth and correction != ground_truth:
                    fp += 1
                if original != ground_truth and correction != ground_truth:
                    fn += 1

                current += 1

            ec_f1 = tp / (tp + (fp + fn) / 2 + 1e-16) # metrics.f1_score(all_y, new_pred, zero_division=0)
            # print(tp, fp, fn)
            return f1, p, r, ec_f1
        else:
            return f1, p, r
    else:
        best_th = 0.5
        p = r = f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred, zero_division=0)
            new_p = metrics.precision_score(all_y, pred, zero_division=0)
            new_r = metrics.recall_score(all_y, pred, zero_division=0)

            if new_f1 > f1:
                f1 = new_f1
                p = new_p
                r = new_r
                best_th = th

        return f1, p, r, best_th



def create_batches(u_set, batch_size, n_ssl_epochs, num_clusters=50):
    """Generate batches such that similar entries are grouped together.
    """
    N = len(u_set)
    tfidf = TfidfVectorizer().fit_transform(u_set.instances)

    kmeans = KMeans(n_clusters=num_clusters).fit(tfidf)

    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx)

    # report FNR within clusters
    if u_set.ground_truth is not None:
        total = 0
        matches = 0
        for cluster in clusters:
            for idx1 in cluster:
                for idx2 in cluster:
                    if idx1 == idx2:
                        continue
                    total += 1
                    if (idx1, idx2) in u_set.ground_truth:
                        matches += 1
        mlflow.log_metric("clustering_FNR", matches / total)

    # concatenate
    for _ in range(n_ssl_epochs):
        indices = []
        random.shuffle(clusters)

        for c in clusters:
            random.shuffle(c)
            indices += c

        batch = []
        for i, idx in enumerate(indices):
            batch.append(u_set[i])
            if len(batch) == batch_size or i == N - 1:
                yield u_set.pad(batch)
                batch.clear()


def selfsl_step(train_nolabel_iter, train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_nolabel_iter (Iterator): the unlabeled data loader
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    # train Barlow Twins or SimCLR
    for i, batch in enumerate(train_nolabel_iter):
        yA, yB = batch
        optimizer.zero_grad()
        # loss = model(i%2, yA, yB, [], da=hp.da)
        if hp.ssl_method == 'simclr':
            # simclr
            loss = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        elif hp.ssl_method == 'barlow_twins':
            # barlow twins
            loss = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        else:
            # combined
            alpha = 1 - hp.alpha_bt
            loss1 = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            loss2 = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            loss = alpha * loss1 + (1 - alpha) * loss2

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"    step: {i}, loss: {loss.item()}")
        #print(f"    step: {i}, loss: {loss.item()}")
        del loss


def fine_tune_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if len(batch) == 4:
            x1, x2, x12, y = batch
            prediction = model(2, x1, x2, x12)
        else:
            x, y = batch
            prediction = model(2, x, None, None)

        loss = criterion(prediction, y.to(model.device))
        # loss = criterion(prediction, y.float().to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"    fine tune step: {i}, loss: {loss.item()}")
        del loss


def train(trainset_nolabel, trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DMDataset): the training set
        validset (DMDataset): the validation set
        testset (DMDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    if not hp.clustering:
        train_nolabel_iter = data.DataLoader(dataset=trainset_nolabel,
                                             batch_size=hp.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=trainset_nolabel.pad)
    else:
        train_nolabel_iter = create_batches(trainset_nolabel,
                                            hp.batch_size,
                                            hp.n_ssl_epochs,
                                            num_clusters=hp.num_clusters)

    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size//2,    # half of barlow twins'
                                 shuffle=True, # TODO: do not shuffle for data cleaning task
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=validset.pad)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=testset.pad)

    # if all_pairs.txt is avaialble
    all_pairs_path = 'data/%s/%s/all_pairs.txt' % (hp.task_type, hp.task)
    if os.path.exists(all_pairs_path):
        all_pair_set = DMDataset(all_pairs_path,
                         lm=hp.lm,
                         size=None,
                         max_len=hp.max_len)

        all_pairs_iter = data.DataLoader(dataset=all_pair_set,
                                     batch_size=hp.batch_size*16,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=testset.pad)
    else:
        all_pairs_iter = None

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # barlow twins
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)

    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        opt_level = 'O2' if hp.ssl_method == 'combined' else 'O2'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # number of steps
    num_ssl_epochs = hp.n_ssl_epochs
    num_ssl_steps = len(trainset_nolabel) // hp.batch_size * num_ssl_epochs
    num_finetune_steps = len(trainset) // (hp.batch_size // 2) * (hp.n_epochs - num_ssl_epochs)
    if num_finetune_steps < 0:
        num_finetune_steps = 0
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_ssl_steps+num_finetune_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    start_epoch = 1

    # load checkpoint if saved
    if hp.use_saved_ckpt:
        ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
        # config_path = os.path.join(hp.logdir, hp.task, 'config.json')
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(start_epoch, hp.n_epochs+1):
        # bootstrap the training set
        # if epoch == num_ssl_epochs + 1 and hp.task_type in ['em', 'cleaning']:
        if epoch == num_ssl_epochs + 1 and hp.task_type in ['em'] and (hp.bootstrap or hp.zero):
            if hp.task_type == 'em':
                new_trainset, TPR, TNR, FPR, FNR = bootstrap(model, hp)
                # logging
                writer.add_scalars(run_tag,
                                   {'new_size': len(new_trainset),
                                    'TPR': TPR,
                                    'TNR': TNR,
                                    'FPR': FPR,
                                    'FNR': FNR}, epoch)

                new_size = len(new_trainset)
                for variable in ["new_size", "TPR", "TNR", "FPR", "FNR"]:
                    mlflow.log_metric(variable, eval(variable))
            elif hp.task_type == 'cleaning':
                new_trainset = bootstrap_cleaning(model, hp)

            train_iter = data.DataLoader(dataset=new_trainset,
                                         batch_size=hp.batch_size//2,    # half of barlow twins'
                                         shuffle=True,
                                         num_workers=0,
                                         collate_fn=padder)

        # train
        print(f"epoch {epoch}")
        model.train()
        if epoch <= num_ssl_epochs:
            selfsl_step(train_nolabel_iter, train_iter, model, optimizer, scheduler, hp)
            if hp.blocking:
                recall, new_size = evaluate_blocking(model, hp)
                # logging
                if isinstance(recall, list):
                    scalars = {}
                    for i in range(len(recall)):
                        scalars['recall_%d' % i] = recall[i]
                        scalars['new_size_%d' % i] = new_size[i]
                else:
                    scalars = {'recall': recall,
                               'new_size': new_size}
                writer.add_scalars(run_tag, scalars, epoch)
                for sz, r in zip(new_size, recall):
                    mlflow.log_metric("recall_%d" % sz, r)

                # for variable in ["new_size", "recall"]:
                #     mlflow.log_metric(variable, eval(variable))
                print(f"epoch {epoch}: recall={recall}, num_candidates={new_size}")
        else:
            fine_tune_step(train_iter, model, optimizer, scheduler, hp)

            # eval
            model.eval()
            dev_f1, dev_p, dev_r, th = evaluate(model, valid_iter)
            if hp.task_type == 'cleaning':
                # cleaning
                test_f1, test_p, test_r, ec_f1 = evaluate(model, test_iter,
                                          threshold=th,
                                          ec_task=hp.task,
                                          dump=True)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_test_f1 = test_f1
                    best_ec_f1 = ec_f1
                print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}, ec_f1={ec_f1}")

                # logging
                scalars = {'f1': dev_f1,
                           'p': dev_p,
                           'r': dev_r,
                           't_f1': test_f1,
                           't_p': test_p,
                           't_r': test_r,
                           'ec_f1': ec_f1}
                for variable in ["dev_f1", "dev_p", "dev_r", "test_f1", "test_p", "test_r", "ec_f1"]:
                    mlflow.log_metric(variable, eval(variable))
            else:
                # em
                test_f1, test_p, test_r = evaluate(model, test_iter, threshold=th, dump=True)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_test_f1 = test_f1
                print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")
                print(f"epoch {epoch}: dev_p={dev_p}, dev_r={dev_r}, test_p={test_p}, test_r={test_r}")

                # run on all pairs
                if epoch == hp.n_epochs and all_pairs_iter is not None:
                    evaluate(model, all_pairs_iter, threshold=th, dump=True)
                    # evaluate(model, test_iter, threshold=th, dump=True)

                # logging
                scalars = {'f1': dev_f1,
                           'p': dev_p,
                           'r': dev_r,
                           't_f1': test_f1,
                           't_p': test_p,
                           't_r': test_r}
                for variable in ["dev_f1", "dev_p", "dev_r", "test_f1", "test_p", "test_r"]:
                    mlflow.log_metric(variable, eval(variable))
            writer.add_scalars(run_tag, scalars, epoch)

        # saving checkpoint at the last ssl step
        if hp.save_ckpt and epoch == num_ssl_epochs:
            # create the directory if not exist
            directory = os.path.join(hp.logdir, hp.task)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # save the checkpoints for each component
            ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
            config_path = os.path.join(hp.logdir, hp.task, 'config.json')
            ckpt = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}
            torch.save(ckpt, ckpt_path)

        # check if learning rate drops to 0
        if scheduler.get_last_lr()[0] < 1e-9:
            break

    writer.close()
