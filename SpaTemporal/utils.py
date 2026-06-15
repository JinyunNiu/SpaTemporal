#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/11/25 14:40 
# @Author  : niujinyun
# @File    : utils.py
# @Email   : niujinyun@aliyun.com
# cython: language_level=3

import os
import time
import torch
import random
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from torch.backends import cudnn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, contingency_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def fix_seed(seed):
    # seed = 2025
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def hclust_dendrogram(
    adata,
    num_cluster,
    use_rep='emb_pca',
    method='ward',
    metric='euclidean',
    random_state=2025,
    plot=True
):
    np.random.seed(random_state)
    data = adata.obsm[use_rep]

    Z = linkage(data, method=method, metric=metric)

    labels = fcluster(Z, num_cluster, criterion='maxclust')

    adata.obs['hclust'] = labels
    adata.obs['hclust'] = adata.obs['hclust'].astype('int')
    adata.obs['hclust'] = adata.obs['hclust'].astype('category')

    if plot:
        plt.figure(figsize=(12, 6))
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=num_cluster,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=True
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Cluster index")
        plt.ylabel("Distance")
        plt.show()

    return adata

def hclust(adata, num_cluster, used_obsm='emb_pca', method='ward', metric='euclidean', random_state=2025):
    np.random.seed(random_state)

    data = adata.obsm[used_obsm]

    clustering = AgglomerativeClustering(
        n_clusters=num_cluster,
        metric=metric,
        linkage=method
    )
    labels = clustering.fit_predict(data)

    adata.obs['hclust'] = labels
    adata.obs['hclust'] = adata.obs['hclust'].astype('int')
    adata.obs['hclust'] = adata.obs['hclust'].astype('category')

    return adata


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', key_added='mclust', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def Kmeans(adata, num_cluster, used_obsm='emb_pca', key_added='kmeans', init='k-means++', n_init=10, max_iter=300, random_seed=2025):

    data = adata.obsm[used_obsm]
    kmeans = KMeans(num_cluster, init=init, n_init=n_init, max_iter=max_iter, random_state=random_seed).fit_predict(data)

    adata.obs[key_added] = kmeans
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'Method', "DLPFC"])

    ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, ari, methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, nmi, methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    hs = homogeneity_score(pred, gt)
    df = df.append(pd.Series([sample, hs, methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, purity, methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    return df

def refine(
    sample_id,
    pred,
    dis,
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)

    return refined_pred


#time decorator
def get_running_time(func):
    def func_time(*args, **kwargs):
        t0 = time.time()
        print(f"{get_format_time()} Method: '{func.__name__}' Running...")
        res = func(*args, **kwargs)
        t1 = time.time()
        print(f"  Running time: {((t1 - t0) // 60)} min {((t1 - t0) % 60):.4f} s")
        return res

    return func_time

def get_format_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


#memory usage
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_usages = result.strip().split('\n')
        for idx, usage in enumerate(gpu_usages):
            used, total = usage.split(', ')
            print(f"GPU {idx}: {used} MiB / {total} MiB")
    except Exception as e:
        print("Could not query GPU memory usage:", e)


#plot aligned slices
import matplotlib
from matplotlib import cm

def plot_aligned_slices(adata, coor_key="loc_scaled", title="After alignment"):
    cmap = cm.get_cmap('rainbow', len(adata.obs["timepoint_numeric"].unique()))
    colors_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(adata.obs["timepoint_numeric"].unique()))]

    plt.figure(figsize=(9, 6))
    plt.title(title)
    for i in range(len(adata.obs["timepoint_numeric"].unique())):
        coords = adata[adata.obs["timepoint_numeric"]==i].obsm[coor_key]
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=colors_list[i],
                    label=f"Slice {i}",
                    s=5., alpha=0.5)

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=(1.02, 0.2), ncol=(len(adata.obs["timepoint_numeric"].unique())//13 + 1))
    plt.tight_layout()
    plt.show()



