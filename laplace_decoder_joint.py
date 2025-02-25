

import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import matplotlib.pyplot as plt
import os
from basemodel import MLP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super(PositionalEncoding, self).__init__()

        positional_encodings = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
                
        self.register_buffer('positional_encodings', positional_encodings.unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encodings[:, :x.size(1), :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.pe = PositionalEncoding(d_model)
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*2, d_model),
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # x = self.pe(x)

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-1e8'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        context = torch.matmul(attn_probs, values)

        # Concatenate heads and apply final output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_projection(context)

        
        return output


class Decoder(nn.Module):

    def __init__(self, args) -> None:
        super(Decoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length

        self.num_modes = 20
        self.min_scale = min_scale
        self.args = args
        
        self.lstm = nn.LSTMCell(input_size=self.hidden_size,
                          hidden_size=self.hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=self.hidden_size,
                          hidden_size=self.hidden_size)
        self.self_attention = MultiHeadSelfAttention(self.hidden_size, 4)
        # self.self_attention = nn.MultiheadAttention(self.hidden_size, 4, batch_first=True)
        


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
            
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))

        

        self.apply(init_weights)

    # def cal_adj_matrix(self, coordinates, batch_split):
    #     # [N,2]
    #     N = coordinates.size(0)
    #     with torch.no_grad():
    #         adj_matrix = torch.zeros(N, N, device=coordinates.device)
    #         for left, right in batch_split:
    #             pre_gen_traj = coordinates[left:right] # [n,2]
    #             pre_tmp = pre_gen_traj.transpose(-1,-2) # [2,n]
    #             distance = torch.abs(pre_tmp.unsqueeze(-1) - pre_tmp.unsqueeze(-2)) # [2,n,n] 计算邻接矩阵
    #             adj_matrix[left:right, left:right] = (distance[:, 0] < 10) & (distance[:, 1] < 10) # [n,n]
    #     return adj_matrix # [N, N]

    def forward(self, global_embed, hidden_state, cn, shift_values, max_values, batch_split):
        dev = global_embed.device
        shift_values = shift_values.squeeze(0)

        
        N = hidden_state.size(0)
        global_embed = self.multihead_proj_global(global_embed).view(12, -1, self.num_modes, self.hidden_size)  # [H, N, F, D]
        global_embed = global_embed.transpose(1,2)  # [H, F, N, D]

        local_embed = hidden_state.repeat(self.num_modes, 1, 1) # [20, N, D]
        cn = cn.repeat(self.num_modes, 1, 1)

        

        global_embed_1 = global_embed.reshape(self.future_steps, -1, self.hidden_size) # [12,20N,D]
        hn_1 = local_embed.reshape(-1, self.hidden_size) # [20N,D]
        cn_1 = cn.reshape(-1, self.hidden_size)
        
        
        output1_list = []
        for t in range(self.future_steps):
            hn_1, cn_1 = self.lstm(global_embed_1[t], (hn_1, cn_1)) # [20N,D]
            output1_list.append(hn_1)
        output1 = torch.stack(output1_list) # [12,20N,D]
        output1_t = output1.transpose(0, 1) # [20N,12,D]
        loc1 = self.loc(output1_t).view(self.num_modes, -1, self.future_steps, 2) # [20, N, 12, 2]
        scale1 = F.elu_(self.scale(output1_t), alpha=1.0) + 1.0 + self.min_scale
        scale1 = scale1.view(self.num_modes, -1, self.future_steps, 2)
        


        # future interaction
        # 还原坐标
        loc1_ = loc1 * max_values.unsqueeze(-2) + shift_values.unsqueeze(-2) # [20,12,2,N]
        loc1_ = loc1_.transpose(1,2).reshape(self.num_modes*self.future_steps, -1, 2).transpose(-1,-2) # [20*12,2,N]
        distance_whole = torch.abs(loc1_.unsqueeze(-1) - loc1_.unsqueeze(-2)) # [20*12,2,N,N]
        # social_mask = torch.zeros(self.num_modes*self.future_steps, loc1.size(1), loc1.size(1), device=dev) # [20*12,N,N]
        modal_mask = torch.ones(self.num_modes, self.num_modes, device=dev) # [20,20]
        social_modal_future = torch.zeros(self.future_steps, self.num_modes, N, self.hidden_size, device=dev) # [12,20,N,D]

        output1_tmp = output1.detach()
        output1_tmp = output1_tmp.reshape(self.future_steps, self.num_modes, -1, self.hidden_size) # [12,20,N,D]
        for left, right in batch_split:
            now_n = right - left
            
            distance_b = distance_whole[:, :, left:right, left:right] # [20*12,n,n]
            social_mask_b = (distance_b[:, 0] < 10) & (distance_b[:, 1] < 10) # [20*12,n,n]
            social_mask_b = social_mask_b.reshape(self.num_modes, self.future_steps, now_n, now_n) # [20,12,n,n]
            social_mask_full_b_exist = (social_mask_b).any(0)
            # social_mask_full_b = ~social_mask_full_b_exist

            joint_mask_b = social_mask_full_b_exist.repeat(1, 20, 20).bool() # [12, 20n,20n]

            output1_tmp_b = output1_tmp[:, :, left:right, :].reshape(self.future_steps, -1, self.hidden_size) # [12,20n,D]
            social_modal_future_b = self.self_attention(output1_tmp_b, joint_mask_b) # [12,20n,D]
            social_modal_future_b = social_modal_future_b.reshape(self.future_steps, self.num_modes, now_n, self.hidden_size) # [12,20,n,D]
            social_modal_future[:, :, left:right, :] = social_modal_future_b # [12,20,n,D]

        social_modal_future = social_modal_future.reshape(self.future_steps, -1, self.hidden_size) # [12,20N,D]



        global_embed_2 = global_embed_1
        output2_list = []
        hn_2 = local_embed.reshape(-1, self.hidden_size) # [20N,D]
        cn_2 = cn.reshape(-1, self.hidden_size)
        for t in range(self.future_steps):
            cn_2 = cn_2 + social_modal_future[t]
            hn_2 = hn_2 + torch.tanh(cn_2)
            hn_2, cn_2 = self.lstm2(global_embed_2[t], (hn_2, cn_2)) # [20N,D]
            output2_list.append(hn_2)
        output2 = torch.stack(output2_list)
        output2_t = output2.transpose(0, 1) # [20N,12,D]
        
        loc2 = self.loc(output2_t).view(self.num_modes, -1, self.future_steps, 2) # [20, N, 12, 2]
        scale2 = F.elu_(self.scale(output2_t), alpha=1.0) + 1.0 + self.min_scale
        scale2 = scale2.view(self.num_modes, -1, self.future_steps, 2)

        return (loc1, scale1), (loc2, scale2)


class MLPDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(MLPDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        # self.input_size = self.args.hidden_size + self.args.z_dim
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))   
        self.apply(init_weights)

    def forward(self, x_encode: torch.Tensor, hidden_state, cn) -> Tuple[torch.Tensor, torch.Tensor]:
        x_encode = self.multihead_proj_global(x_encode).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
        x_encode = x_encode.transpose(0, 1)  # [F, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(torch.cat((local_embed, x_encode), dim=-1)).squeeze(-1).t()  # [N, F]
        out = self.aggr_embed(torch.cat((x_encode, local_embed), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
        scale = scale + self.min_scale  # [F, N, H, 2]
        return (loc, scale, pi) # [F, N, H, 2], [F, N, H, 2], [N, F]
   
    def plot_pred(self, loc, lock, N=10, groundtruth=True):
        """
        This is the plot function to plot the first scene
        lock:   [N, K, H, 2]
        loc: [N, F, H, 2]
        """
        
        fig,ax = plt.subplots()
        pred_seq = loc.shape[2]
        lock = lock.cpu().detach().numpy()
        loc = loc.cpu().detach().numpy()
        for m in range(loc.shape[0]):
            for i in range(loc.shape[1]):
                y_p_sum = np.cumsum(loc[m,i,:,:], axis=0)
                ax.plot(y_p_sum[:, 0], y_p_sum[:, 1], color='k', linewidth=1)
            for j in range(lock.shape[1]):
                y_sum = np.cumsum(lock[m,j,:,:], axis=0)
                ax.plot(y_sum[:, 0], y_sum[:, 1], color='r', linewidth=3)
            ax.set_aspect("equal")
            path = "plot/kmeans++"
            if not os.path.exists(path):
                os.mkdir(path) 
            plt.savefig(path+"/"+str(len(os.listdir(path)))+".png")
            print(path + "/" +str(len(os.listdir(path)))+".png")
            plt.gcf().clear()
            plt.close() 
            
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
