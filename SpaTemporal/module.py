#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/11/25 13:40 
# @Author  : niujinyun
# @File    : module.py
# @Email   : niujinyun@aliyun.com
# cython: language_level=3

import copy
import numpy as np
import random
import torch.nn.functional as F
import torch

from functools import partial
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

# def sce_loss(x, y, t=2):
#     x = F.normalize(x, p=2, dim=-1)
#     y = F.normalize(y, p=2, dim=-1)
#     cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
#     loss = -torch.log(cos_m.pow_(t))
#     return loss.mean()

def create_activation(name=None):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "lkrelu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
    
def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        act,
        nn.Dropout(p=p_drop),
    )

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat": 
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x

# class Discriminator(nn.Module):
#     def __init__(self, n_h):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#         c_x = c.expand_as(h_pl)  

#         sc_1 = self.f_k(h_pl, c_x)
#         sc_2 = self.f_k(h_mi, c_x)

#         if s_bias1 is not None:
#             sc_1 += s_bias1
#         if s_bias2 is not None:
#             sc_2 += s_bias2

#         logits = torch.cat((sc_1, sc_2), 1)

#         return logits
    
# class AvgReadout(nn.Module):
#     def __init__(self):
#         super(AvgReadout, self).__init__()

#     def forward(self, emb, mask=None):
#         vsum = torch.mm(mask, emb)
#         row_sum = torch.sum(mask, 1)
#         row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
#         global_emb = vsum / row_sum 
          
#         return F.normalize(global_emb, p=2, dim=1)

class module(nn.Module): 
    def __init__(self, 
                 input_dim,
                 #graph_neigh,
                 feat_hidden1=64,
                 feat_hidden2=16,
                 gcn_hidden=64,
                 latent_dim=16,
                 p_drop=0.1,
                 mask_rate=0.2,
                 act_name='relu',
                 mode='zero'
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden = gcn_hidden
        self.latent_dim = latent_dim
        self.mask_rate = mask_rate
        self.p_drop = p_drop
        #self.graph_neigh = graph_neigh
        self.act = create_activation(act_name)
        self.mode = mode

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))
        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

        self.decoder = GraphConv(self.latent_dim, self.input_dim, dropout=self.p_drop, act=nn.Identity())

        self.criterion = self.setup_loss_fn(loss_fn='sce')
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        # self.disc = Discriminator(self.latent_dim)
        # self.sigm = nn.Sigmoid()
        # self.read = AvgReadout()

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss)
        else:
            raise NotImplementedError
        return criterion
        
    
    def encode(self, x, edge_index):
        x = self.encoder(x)
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return x

    def decode(self, x, edge_index):
        x = self.decoder(x, edge_index)
        return x


    def mask_noise(self, x, adj, mask_rate=0.2, mode='zero'):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        if mask_rate > 0:
            num_mask_nodes = int(mask_rate * num_nodes) 
            mask_nodes = perm[:num_mask_nodes]  
            keep_nodes = perm[num_mask_nodes:]  

            out_x = x.clone()  
            token_nodes = mask_nodes
            if mode == 'zero':
                out_x[token_nodes] = 0.0 
            if mode == 'token':
                out_x[token_nodes] += self.enc_mask_token
            else:
                raise ValueError(
                    f"Unsupported mask mode: '{mode}'. 'mode' must be 'zero' or 'token'."
                )
            use_adj = adj.clone()

            return out_x, use_adj, (mask_nodes, keep_nodes)
        else:
            return x, adj, (torch.tensor([]).to(x.device), torch.tensor([]).to(x.device))
        
        
    def forward(self, x, edge_index):
        x, edge_index, (mask_nodes, keep_nodes) = self.mask_noise(x, edge_index, self.mask_rate, self.mode)
        z = self.encode(x, edge_index)
        x_rec = self.decode(z, edge_index)

        recon = x_rec.clone()  
        x_init = x[mask_nodes]  
        x_recon = recon[mask_nodes]

        loss_self = self.criterion(x_recon, x_init)

        # emb = self.act(z)

        # z_a = self.encode(x_a, edge_index)
        # emb_a = self.act(z_a)

        # g = self.read(emb, self.graph_neigh) 
        # g = self.sigm(g) 

        # g_a = self.read(emb_a, self.graph_neigh)
        # g_a = self.sigm(g_a) 

        # ret = self.disc(g, emb, emb_a)  
        # ret_a = self.disc(g_a, emb_a, emb) 
        
        #return z, x_rec, loss_self, ret, ret_a
        return z, x_rec, loss_self

        



