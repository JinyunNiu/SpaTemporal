#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/11/25 21:40
# @Author  : niujinyun
# @File    : graph_3D.py
# @Email   : niujinyun@aliyun.com
# cython: language_level=3

import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F

from torch import nn
from sklearn.cluster import KMeans
from module import module
from tqdm import tqdm

from mnn import *
from utils import get_running_time, mclust_R, KMeans
from process import get_feature, add_contrastive_label, permutation


def reconstruction_loss(x_rec, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(x_rec, x)
    return loss_rcn

class Train(torch.nn.Module):
    def __init__(self, 
                 adata, 
                 graph_dict, 
                 rec_weight=10, 
                 self_weight=0.1,
                 sl_weight=0.1,
                 tri_weight=1,
                 mask_rate=0.2, 
                 mode='zero',
                 pre_epochs=500,
                 epochs=500,
                 lr=0.001,
                 decay=0.001,
                 stride=100,
                 num_cluster=10,\
                 grad_down=5,
                 device='cuda:0',
                 ):
        super(Train, self).__init__()
        self.device = device
        self.adata = adata.copy()
        self.X = self.adata.obsm['X_pca']
        self.X = torch.FloatTensor(self.X.copy()).to(self.device)
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.norm_value = graph_dict["norm_value"]
        self.rec_weight = rec_weight
        self.self_weight = self_weight
        self.sl_weight = sl_weight
        self.tri_weight = tri_weight
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.stride = stride
        self.num_cluster = num_cluster
        self.grad_down = grad_down

        # if 'label_CSL' not in adata.obsm.keys():    
        #    add_contrastive_label(self.adata)

        # if 'feat' not in adata.obsm.keys():
        #    self.features_a = get_feature(self.features)

        # self.features = torch.FloatTensor(self.features.copy()).to(self.device)
        # self.features_a = torch.FloatTensor(self.features_a.copy()).to(self.device)
        # self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        # self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adata.shape[0])).to(self.device)

        self.model = module(input_dim=self.X.shape[1],
                            #graph_neigh=self.graph_neigh,
                            mask_rate=mask_rate,
                            mode=mode
                            ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)
        
    def train_init_model(
              self, 
              ):
        
        #self.loss_CSL = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in tqdm(range(self.pre_epochs)):
            self.model.train()
            self.optimizer.zero_grad()

            #self.features_a = permutation(self.features)

            #z, x_rec, loss_self, ret, ret_a = self.model(self.features, self.features_a, self.adj_norm)
            z, x_rec, loss_self = self.model(self.X, self.adj_norm)

            #loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            #loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)

            loss_rec = reconstruction_loss(x_rec, self.X)

            #loss = self.rec_weight * loss_rec + self.self_weight * loss_self + self.self_weight * (loss_sl_1 + loss_sl_2)
            loss = self.rec_weight * loss_rec + self.self_weight * loss_self

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_down)
            self.optimizer.step()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    @torch.no_grad()
    def embedding(self):
        self.model.eval()
        # z, x_rec, _, _, _ = self.model(self.features, self.features_a, self.adj_norm)
        z, x_rec, _ = self.model(self.X, self.adj_norm)
        
        latent_z = z.data.cpu().numpy()
        x_recon = x_rec.data.cpu().numpy()
        
        return latent_z, x_recon
    
    def clustering(self, adata, num_cluster, used_obsm, key_added, random_seed=2025, method='mclust'):
        assert method in ['mclust', 'kmeans']
        if method == 'mclust':
            adata = mclust_R(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added=key_added, random_seed=random_seed)
        elif method == 'kmeans':
            adata = KMeans(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added=key_added, random_seed=random_seed)
        else:
            raise Exception
        return adata
    
    def train_final_model(
            self,
            knn_neigh=100,
            margin=1.0,
            iter_comb=None):
        
        self.train_init_model()
        #pre_z, _ = self.embedding()

        # with tqdm(total=int(self.pre_epochs),
        #           desc="Trains a final model",
        #           bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for epoch in tqdm(range(self.epochs)):
            if epoch % self.stride == 0:
                pre_z, _ = self.embedding()
                self.adata.obsm['pre_z'] = pre_z

                self.clustering(self.adata, num_cluster=self.num_cluster, used_obsm='pre_z', key_added="tmp_domain", method='mclust')

                section_ids = np.array(self.adata.obs['timepoint'].unique())
                mnn_dict = create_dictionary_mnn_c(self.adata, use_rep='pre_z', use_label='tmp_domain', batch_name='timepoint', k=knn_neigh,
                                                    iter_comb=iter_comb, verbose=0)

                anchor_ind = []
                positive_ind = []
                negative_ind = []
                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = self.adata.obs['timepoint'][mnn_dict[batch_pair].keys()]

                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = self.adata.obs_names[
                            self.adata.obs['timepoint'] == section_ids[batch_id]].values

                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        ## np.random.choice(mnn_dict[batch_pair][anchor])
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
                    anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                    positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                    negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

            torch.set_grad_enabled(True)
            self.model.train()
            self.optimizer.zero_grad()
            z, x_rec, loss_self = self.model(self.X, self.adj_norm)

            loss_rec = reconstruction_loss(x_rec, self.X)

            anchor_arr = z[anchor_ind,]
            positive_arr = z[positive_ind,]
            negative_arr = z[negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            loss_tri = triplet_loss(anchor_arr, positive_arr, negative_arr)


            loss = self.rec_weight * loss_rec + self.self_weight * loss_self + self.tri_weight * loss_tri

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_down)
            self.optimizer.step()



        




              
        


