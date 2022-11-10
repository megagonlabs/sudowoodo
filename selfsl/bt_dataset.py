import torch
import random

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter
from .utils import blocked_matmul

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class BTDataset(data.Dataset):
    """Dataset for pre-training"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da='all'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.instances= []
        self.max_len = max_len
        self.size = size

        for line in open(path):
            self.instances.append(line.strip())

        if size is not None:
            if size > len(self.instances):
                N = size // len(self.instances) + 1
                self.instances = (self.instances * N)[:size]
            else:
                self.instances = random.sample(self.instances, size)

        self.da = da    # default is random
        self.augmenter = Augmenter()
        self.ground_truth = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.instances)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the 1st entity
            List of int: token ID's of the 2nd entity
            List of int: token ID's of the two entities combined
            int: the label of the pair (0: unmatch, 1: match)
        """
        if self.da == 'cutoff':
            # A = B = self.instances[idx]
            A = self.instances[idx]
            # combine with the deletion operator
            B = self.augmenter.augment_sent(A, "del") 
        else:
            A = self.instances[idx]
            B = self.augmenter.augment_sent(A, self.da)

        # left
        yA = self.tokenizer.encode(text=A,
                                   max_length=self.max_len,
                                   truncation=True)
        yB = self.tokenizer.encode(text=B,
                                   max_length=self.max_len,
                                   truncation=True)
        return yA, yB

    def create_ground_truth(self, datasets):
        """Add ground truths to the unlabeled set for evaluation.

        Args:
            datasets (List of DMDataset): the (train, valid, test) datasets

        Returns:
            None
        """
        mp = {}
        for idx, inst in enumerate(self.instances):
            mp[inst] = idx

        self.ground_truth = set([])
        for dataset in datasets:
            for pair, label in zip(dataset.pairs, dataset.labels):
                if int(label) == 1:
                    left, right = pair
                    left = left.strip()
                    right = right.strip()
                    if left in mp and right in mp:
                        left, right = mp[left], mp[right]
                        self.ground_truth.add((left, right))
                        self.ground_truth.add((right, right))
        print(len(self.ground_truth))

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
        """
        yA, yB = zip(*batch)

        maxlen = max([len(x) for x in yA])
        maxlen = max(maxlen,max([len(x) for x in yB]))

        yA = [xi + [0]*(maxlen - len(xi)) for xi in yA]
        yB = [xi + [0]*(maxlen - len(xi)) for xi in yB]

        return torch.LongTensor(yA), \
               torch.LongTensor(yB)
