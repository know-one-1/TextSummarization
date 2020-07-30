import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize

INI = 1e-2

class ConvDiscrim(nn.Module):
    """
    Convolutional word-level sentence Discriminator
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout=0.01):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i,padding=1)
                                     for i in range(3, 6)])
        self.fc1=nn.Linear(n_hidden*len(range(3,6)),100)
        self.fc2=nn.Linear(100,50)
        self.fc3=nn.Linear(50,2)

        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        conv_out = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        x = F.relu(self.fc1(conv_out))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class CopyConvDiscrim(nn.Module):
    """
    Convolutional word-level sentence Discriminator
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout=0.01):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i,padding=1)
                                     for i in range(3, 6)])
        self._conv2 = nn.ModuleList([nn.Conv1d(len(range(3,6)),n_hidden,i)
                                       for i in range(2,4)])
        self.fc1=nn.Linear(n_hidden*len(range(2,4)),80)
        self.fc2=nn.Linear(80,50)
        self.fc3=nn.Linear(50,2)

        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_1, input_2):
        sents, extrts = input_1,input_2

        emb_sents = self._embedding(sents)
        emb_extrts = self._embedding(extrts)
        
        emb_extrts = emb_extrts if len(emb_extrts.size())==3 else emb_extrts.squeeze()
        emb_sents =  emb_sents if len(emb_sents.size())==3 else emb_sents.squeeze()
        conv_in_sent = F.dropout(emb_sents.transpose(1, 2),
                            self._dropout, training=self.training)
        conv_in_extrt = F.dropout(emb_extrts.transpose(1, 2),
                            self._dropout, training=self.training)
        conv_out_sent = torch.stack([F.relu(conv(conv_in_sent)).max(dim=2)[0]
                            for conv in self._convs], dim=2)
        conv_out_extrt = torch.stack([F.relu(conv(conv_in_extrt)).max(dim=2)[0]
                            for conv in self._convs], dim=2)
        conv_out=torch.cat([conv_out_sent,conv_out_extrt],dim=1)

        conv_out_f= F.dropout(conv_out.transpose(1,2),self._dropout,training=self.training)
        
        ff_in = torch.cat([F.relu(conv(conv_out_f)).max(dim=2)[0] 
                                for conv in self._conv2],dim=1)

        x = F.relu(self.fc1(ff_in))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)