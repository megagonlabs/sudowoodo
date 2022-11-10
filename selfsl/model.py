import torch
import torch.nn as nn

from transformers import AutoModel

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DMModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', task_type='em', pretrained=True):
        super().__init__()
        
        if pretrained:
            path = 'data/em/SSL-baseline/ssl-finetuned/pytorch_model.bin'
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.bert = AutoModel.from_pretrained(lm_mp[lm], state_dict=state_dict)
        else:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])

        self.device = device
        self.task_type = task_type
        hidden_size = 768
        if task_type == 'em':
            self.fc = torch.nn.Linear(hidden_size * 2, 2)
        else:
            self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None, x12=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's of the left entity
            x2 (LongTensor): a batch of ID's of the right entity
            x12 (LongTensor): a batch of ID's of the left+right

        Returns:
            Tensor: binary prediction
        """
        if self.task_type == 'em':
            # em
            x1 = x1.to(self.device) # (batch_size, seq_len)
            x2 = x2.to(self.device) # (batch_size, seq_len)
            x12 = x12.to(self.device) # (batch_size, seq_len)

            # left+right
            enc_pair = self.bert(x12)[0][:, 0, :] # (batch_size, emb_size)

            batch_size = len(x1)
            # left and right
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            # fully connected
            return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
        else:
            x1 = x1.to(self.device) # (batch_size, seq_len)
            return self.fc(self.bert(x1)[0][:, 0, :])

        # batch_size = len(x1)
        # enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
        # enc1 = enc[:batch_size]
        # enc2 = enc[batch_size:]
        # return self.distance(self.fc(enc1), self.fc(enc2)).sigmoid()
        # return self.fc((enc1 - enc2).abs()) # .squeeze() # .sigmoid()
        # return self.fc((enc1 - enc2).abs())


