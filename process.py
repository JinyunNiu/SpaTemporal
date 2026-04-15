#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/11/25 13:40 
# @Author  : niujinyun
# @File    : dataset.py
# @Email   : niujinyun@aliyun.com
# cython: language_level=3

import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def read_adata(adata_name_list, data_root):
    print("Reading data.")
    for adata_name in tqdm(adata_name_list):
        #adata_tmp = sc.read_visium(os.path.join(data_root, adata_name), count_file= f'{adata_name}_filtered_feature_bc_matrix.h5')
        adata_tmp = sc.read_h5ad(os.path.join(data_root, adata_name + '.MOSTA.h5ad'))
        adata_tmp.var_names_make_unique()
        print("adata {} shape: {}".format(adata_name, adata_tmp.X.shape))

        adata_tmp.obs['timepoint'] = adata_name

        ##### Load label, if have
        # df_label = pd.read_csv(data_root / adata_name / 'manual_annotations.txt', sep='\t', header=None, index_col=0)
        # df_label.columns = ['annotation']
        # adata_tmp.obs['annotation'] = df_label['annotation']

        if adata_name == adata_name_list[0]:
            adata = adata_tmp
            adata.obs['timepoint'] = adata_name
        else:
            var_names = adata.var_names.intersection(adata_tmp.var_names)
            adata = adata[:, var_names]
            adata_tmp = adata_tmp[:, var_names]
            adata_tmp.obs['timepoint'] = adata_name
            adata = adata.concatenate(adata_tmp)
    print("Finished reading data.")
    return adata

def data_process(adata, n_top_genes=3000, n_pcs=200, seed=42):
    print("Processing data.")
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable. 
    adata_X = PCA(n_components=n_pcs, random_state=seed).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    print("Finished processing data.")

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL

def get_feature(feat):
    # data augmentation
    feat_a = permutation(feat)
    
    return feat_a