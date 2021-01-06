import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

class MLP_HMM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MLP_HMM, self).__init__()
        feat_dim = kwargs.get("feat_dim", 0)
        left = kwargs.get("left", 0)
        right = kwargs.get("right", 0)
        num_layers = kwargs.get("num_layers", 0)
        hidden_dim = kwargs.get("hidden_dim", 0)
        dropout = kwargs.get("dropout", 0)
        output_dim = kwargs.get("output_dim", 0)

        input_dim = feat_dim * (left + 1 + right)
        
        self.MLP = nn.ModuleList([
            nn.Sequential(            
                nn.Linear(input_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim, momentum=0.05), nn.ReLU(), nn.Dropout(p=dropout)
            )
        ])
        for idx in range(num_layers-1):
            self.MLP.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim, momentum=0.05), nn.ReLU(), nn.Dropout(p=dropout)
                )
            )
        self.HMM = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )
            
    def forward(self, x):
        h = x
        for idx, mlp in enumerate(self.MLP):
            h = mlp(h)
        
        h = self.HMM(h)
        return h

class BiGRU_HMM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BiGRU_HMM, self).__init__()
        feat_dim = kwargs.get("feat_dim", 0)
        num_layers = kwargs.get("num_layers", 0)
        hidden_dim = kwargs.get("hidden_dim", 0)
        dropout = kwargs.get("dropout", 0)
        output_dim = kwargs.get("output_dim", 0)
        
        self.GRU = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, num_layers=num_layers, 
            dropout=dropout, bidirectional=True)

        self.HMM = nn.Sequential(
            nn.Linear(2*hidden_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )
    
    def cat_packed_sequence(self, packed_sequence):
        output, bsizes = rnn.pad_packed_sequence(packed_sequence)
        out_batch = []
        for bidx, cur_size in enumerate(bsizes):
            cur_batch = output[:cur_size, bidx]
            out_batch.append(cur_batch)
        outs = torch.cat(out_batch, dim=0)
        return outs
            
    def forward(self, x):
        # input = [feat tensor]
        assert type(x) == list, "Input must be assigned as list type"

        inp = rnn.pack_sequence(x, enforce_sorted=False)
        
        h, _ = self.GRU(inp)
        o = self.cat_packed_sequence(h)
        out = self.HMM(o)        
        return out
