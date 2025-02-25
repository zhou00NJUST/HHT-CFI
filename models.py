from basemodel import *
from laplace_decoder_joint import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from my_decoder import Conv_Decoder
from torch_geometric.utils import to_dense_adj
from hyper_graph_transformer import HyperGraphTransformer

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class MyTraj(nn.Module):
    def __init__(self, args):
        super(MyTraj, self).__init__()
        self.args = args
        self.Temperal_Encoder = Temperal_Encoder(self.args)
        self.Laplacian_Decoder = Decoder(self.args)

        self.n_social = 1
        self.social = nn.ModuleList()
        for t in range(self.n_social):
            self.social.append( HyperGraphTransformer(args.hidden_size) )

        self.reg_loss = LaplaceNLLLoss(reduction='mean')

        self.to12 = Conv_Decoder(args)

    def forward(self, inputs, edge_pair, epoch, iftest=False):
        
        batch_abs_gt, batch_norm_gt, batch_split, shift_values, max_values = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        device = torch.device(batch_abs_gt.device)


        batch_class = batch_abs_gt[0,:,-1]
        batch_abs_gt = batch_abs_gt[:,:,:2]
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            zeros = torch.zeros(1, train_x.size(1), 2, device=device)
            train_x = torch.cat([zeros, train_x], dim=0)
        elif self.args.input_mix:
            offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
            pad_offset = torch.zeros_like(position, device=device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
        self.pre_obs=batch_norm_gt[1:self.args.obs_length]
        
        x_encoded, hidden_state_unsplited, cn = self.Temperal_Encoder.forward(train_x)  #[N, H, 2], [N, D], [N, D]
        # to12
        x_encoded = self.to12(x_encoded)
        
        
        hidden_state_global = hidden_state_unsplited.clone()
        cn_global = cn.clone()

        for left, right in batch_split:
            left = left.item()
            right = right.item()
            num_now = right - left
            pair_now = edge_pair[(left, right)][0]
            if len(pair_now) != 0:
                edge_pair_now = pair_now.transpose(0, 1).to(device).long() # [2, n]
                # adj_now = to_dense_adj(edge_pair_now, max_num_nodes=num_now).squeeze(0) # [n, n]
                edge_type_now = torch.zeros(edge_pair_now.size(-1), device=device) # 同一类
                node_type_now = torch.ones_like(batch_class[left: right], dtype=torch.long)
                hidden_now = hidden_state_unsplited[left: right] # .view(1, -1, self.args.hidden_size)
                cn_now = cn[left: right] # .view(1, -1, self.args.hidden_size)
                

                for i in range(self.n_social):
                    social_out = self.social[i].forward(hidden_now, node_type_now, edge_pair_now, edge_type_now)
                    cn_now = social_out + cn_now
                    hidden_now = hidden_now + torch.tanh(cn_now)

                hidden_state_global[left: right] = hidden_now
                cn_global[left: right] = cn_now

            else: # 没有边
                pass

        train_y_gt = train_y.permute(0, 2, 1)
        

        mdn_out1, mdn_out2 = self.Laplacian_Decoder.forward(x_encoded, hidden_state_global, cn_global, shift_values, max_values, batch_split)
        


        lnnl1, full_pre_tra1 = self.mdn_loss(train_y_gt, mdn_out1)  #[K, H, N, 2]
        lnnl2, full_pre_tra2 = self.mdn_loss(train_y_gt, mdn_out2)  #[K, H, N, 2]
        
        return (lnnl1, lnnl2), full_pre_tra2

    def mdn_loss(self, y, y_prime):
        batch_size=y.shape[0]
        
        
        out_mu, out_sigma = y_prime
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0) # N个中每一个20中的最小的坐标
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss += self.reg_loss(y_hat_best, y)
        
        loss = reg_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs, sample_k), axis=0))
        return loss, full_pre_tra


