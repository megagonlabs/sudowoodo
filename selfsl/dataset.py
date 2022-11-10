import torch
import random

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class DMDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        for line in open(path):
            LL = line.strip().split('\t')
            self.pairs.append(tuple(LL[:-1]))
            self.labels.append(int(LL[-1]))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

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
        # idx = random.randint(0, len(self.pairs)-1)

        if len(self.pairs[idx]) == 2:
            # em
            left = self.pairs[idx][0]
            right = self.pairs[idx][1]

            # augment if da is set
            if self.da is not None:
                left = self.augmenter.augment_sent(left, self.da)
                right = self.augmenter.augment_sent(right, self.da)

            # left
            x1 = self.tokenizer.encode(text=left,
                                       max_length=self.max_len,
                                       truncation=True)
            # right
            x2 = self.tokenizer.encode(text=right,
                                       max_length=self.max_len,
                                       truncation=True)
            # left + right
            x12 = self.tokenizer.encode(text=left,
                                        text_pair=right,
                                        max_length=self.max_len,
                                        truncation=True)
            return x1, x2, x12, self.labels[idx]
        else:
            # cleaning
            left = self.pairs[idx][0]
            if self.da is not None:
                left = self.augmenter.augment_sent(left, self.da)
            x = self.tokenizer.encode(text=left,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: x12 of shape (batch_size, seq_len').
                        Elements of x12 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 4:
            # em
            x1, x2, x12, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]

            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]

            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(x12), \
                   torch.LongTensor(y)
        else:
            # cleaning
            x1, y = zip(*batch)
            maxlen = max([len(x) for x in x1])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            return torch.LongTensor(x1), torch.LongTensor(y)
