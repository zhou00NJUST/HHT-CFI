

import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch import nn, Tensor

import matplotlib.pyplot as plt
import os
from basemodel import MLP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np



class GRUDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length

        self.num_modes = 20
        self.min_scale = min_scale
        self.args = args
        self.lstm1 = nn.LSTM(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
                          
                          
        


        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))

        

        self.apply(init_weights)



    def forward(self, global_embed, hidden_state, cn, batch_split):
        dev = global_embed.device
        global_embed = self.multihead_proj_global(global_embed).view(12, -1, self.num_modes, self.hidden_size)  # [H, N, F, D]
        global_embed = global_embed.transpose(1,2)  # [H, F, N, D]

        local_embed = hidden_state.repeat(self.num_modes, 1, 1) # [20, N, D]
        cn = cn.repeat(self.num_modes, 1, 1)
        

        pi = self.pi(torch.cat((local_embed, global_embed[-1, :, :]), dim=-1)).squeeze(-1).t()  # [N, F]


        

        global_embed_1 = global_embed.reshape(self.future_steps, -1, self.hidden_size) # [12,20N,D]
        hn_1 = local_embed.reshape(1, -1, self.hidden_size) # [1,20N,D]
        cn_1 = cn.reshape(1, -1, self.hidden_size)

        output1, (out_hn1, out_cn1) = self.lstm1(global_embed_1, (hn_1, cn_1)) # [12,20N,D] [1,20N,D]
        
        
        # skip
        output1 = output1.transpose(0, 1)

        
        
        loc = self.loc(output1).view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        scale = F.elu_(self.scale(output1), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        
        return (loc, scale, pi)




def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
