#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/11/25 13:40 
# @Author  : niujinyun
# @File    : graph_3D.py
# @Email   : niujinyun@aliyun.com
# cython: language_level=3

import os
import torch

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from process import read_adata

def spatiotemporal_graph(
               adata_name_list,
               data_root,
               scale_factor=1.5,  # scale factor for spatial coordinates
               rad_cutoff=None,  # cutoff radius of spots for building graph
               k_cutoff=12,
               mode='KNN',
               norm=True,
               ):
    if data_root is None:
        raise ValueError("Please provide 'data_root' for reading data!")
    adata = read_adata(adata_name_list, data_root)
    # adata_st = ad.concat(adata_st_list, label="slice_name", keys=section_ids)
    # adata_st.obs['Ground Truth'] = adata_st.obs['Ground Truth'].astype('category')
    # adata_st.obs["slice_name"] = adata_st.obs["slice_name"].astype('category')
    # Build a graph for spots across multiple slices

    adata.obs['timepoint'] = pd.Categorical(adata.obs['timepoint'], categories=adata_name_list, ordered=True)
    adata.obs['timepoint_numeric'] = adata.obs['timepoint'].cat.codes
    loc_tp = np.array(adata.obs['timepoint_numeric']).astype('int')

    loc_xy = adata.obsm['spatial']
    loc_xy_scaled = np.zeros(loc_xy.shape, dtype=np.float64)
    for i in range(len(adata_name_list)):
        scaler = StandardScaler()
        loc_xy_tmp = loc_xy[loc_tp == i, :]
        loc_xy_tmp = scaler.fit_transform(loc_xy_tmp)
        loc_xy_scaled[loc_tp == i, :] = loc_xy_tmp

    loc_tp = loc_tp[:, np.newaxis]
    loc = np.concatenate((loc_xy_scaled, loc_tp), axis=1)
    loc = pd.DataFrame(loc)
    loc.index = adata.obs.index
    loc.columns = ['x', 'y', 'z']

    if rad_cutoff is None:
        # The first adata in adata_list is used as a reference for computing cutoff radius of spots
        loc_ref = loc[loc['z']==0]
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]
        rad_cutoff = min_dist_ref * scale_factor
        loc_tp_scaled = np.zeros(loc.shape[0])
        for i in range(len(adata_name_list)):
            loc_tp_scaled[loc['z']==i] = rad_cutoff * i
    else:
        loc_tp_scaled = loc['z'] * rad_cutoff

    loc_scaled = np.concatenate([loc_xy_scaled, loc_tp_scaled.values.reshape(-1, 1)], axis=1)
    loc_scaled = pd.DataFrame(loc_scaled, index=loc.index, columns=['x', 'y', 'z'])
    adata.obsm['loc_scaled'] = loc_scaled
    
    if mode == 'KNN':
        nbrs = NearestNeighbors(n_neighbors=k_cutoff).fit(loc_scaled)
        distances, indices = nbrs.kneighbors(loc_scaled)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    else:
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(loc_scaled)
        distances, indices = nbrs.radius_neighbors(loc_scaled, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    Spatial_Net = pd.concat(KNN_list)
    Spatial_Net.columns = ['Cell1', 'Cell2', 'Distance']
    #Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(loc_scaled.shape[0]), np.array(loc_scaled.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
    print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    #
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(cells_id_tran)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(cells_id_tran)

    #Spatial_Net = Spatial_Net.drop_duplicates(subset=['Cell1', 'Cell2'], keep='first')
    
    adj_m1 = sp.coo_matrix((np.ones(Spatial_Net.shape[0]), (Spatial_Net['Cell1'], Spatial_Net['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # Store original adjacency matrix (without diagonal entries) for later
    # adj_m1 = adj_m1 + adj_m1.T
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    if norm:
        adj_norm_m1 = preprocess_graph(adj_m1)
    else:
        adj_norm_m1 = sparse_mx_to_torch_sparse_tensor(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])

    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    #norm_m1 = adj_m1.shape[0]**2 / float((adj_m1.shape[0]**2 - adj_m1.sum()))

    # # generate random mask
    # adj_mask = mask_generator(adj_label_m1.to_sparse(), N)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1,
        # "mask": adj_mask
    }
    
    return adata, graph_dict


def spatiotemporal_graph_(
               adata_name_list,
               data_root,
               scale_factor=1.5,  # scale factor for spatial coordinates
               rad_cutoff=None,  # cutoff radius of spots for building graph
               k_cutoff=12,
               mode='KNN',
               norm=True, 
               dmax=50
               ):
    if data_root is None:
        raise ValueError("Please provide 'data_root' for reading data!")
    adata = read_adata(adata_name_list, data_root)
    # adata_st = ad.concat(adata_st_list, label="slice_name", keys=section_ids)
    # adata_st.obs['Ground Truth'] = adata_st.obs['Ground Truth'].astype('category')
    # adata_st.obs["slice_name"] = adata_st.obs["slice_name"].astype('category')
    # Build a graph for spots across multiple slices

    adata.obs['timepoint'] = pd.Categorical(adata.obs['timepoint'], categories=adata_name_list, ordered=True)
    adata.obs['timepoint_numeric'] = adata.obs['timepoint'].cat.codes
    loc_tp = np.array(adata.obs['timepoint_numeric']).astype('int')

    loc_xy = adata.obsm['spatial']
    loc_xy_scaled = np.zeros(loc_xy.shape, dtype=np.float64)
    for i in range(len(adata_name_list)):
        scaler = StandardScaler()
        loc_xy_tmp = loc_xy[loc_tp == i, :]
        loc_xy_tmp = scaler.fit_transform(loc_xy_tmp)
        loc_xy_scaled[loc_tp == i, :] = loc_xy_tmp

    loc_tp = loc_tp[:, np.newaxis]
    loc = np.concatenate((loc_xy_scaled, loc_tp), axis=1)
    loc = pd.DataFrame(loc)
    loc.index = adata.obs.index
    loc.columns = ['x', 'y', 'z']

    if rad_cutoff is None:
        # The first adata in adata_list is used as a reference for computing cutoff radius of spots
        loc_ref = loc[loc['z']==0]
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]
        rad_cutoff = min_dist_ref * scale_factor
        loc_tp_scaled = np.zeros(loc.shape[0])
        for i in range(len(adata_name_list)):
            loc_tp_scaled[loc['z']==i] = rad_cutoff * i
    else:
        loc_tp_scaled = loc['z'] * rad_cutoff

    loc_scaled = np.concatenate([loc_xy_scaled, loc_tp_scaled.values.reshape(-1, 1)], axis=1)
    loc_scaled = pd.DataFrame(loc_scaled, index=loc.index, columns=['x', 'y', 'z'])
    adata.obsm['loc_scaled'] = loc_scaled

    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=k_cutoff-1, spatial="loc_scaled")
        # adj_m1 = graph_computing(adata.obsm['spatial'], n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax, spatial="loc_scaled")
    adj_m1 = sp.coo_matrix(adj_m1)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    if norm:
        adj_norm_m1 = preprocess_graph(adj_m1)
    else:
        adj_norm_m1 = sparse_mx_to_torch_sparse_tensor(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    # adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())


    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    # # generate random mask
    # adj_mask = mask_generator(adj_label_m1.to_sparse(), N)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
    }
    
    return adata, graph_dict

##### generate n
def generate_adj_mat(adata, include_self=False, n=6, spatial="spatial"):
    from sklearn import metrics
    assert spatial in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm[spatial])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adata.obsm['graph_neigh'] = adj_mat

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat

def generate_adj_mat_1(adata, max_dist, spatial="spatial"):
    from sklearn import metrics
    assert spatial in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm[spatial], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


