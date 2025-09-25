# Code adapted from scNODE: https://github.com/rsinghlab/scNODE
# 
# Extended with additional parts for Veres.

import numpy as np
import scanpy
import pandas as pd
import natsort
import torch
import torch.distributions as dist

# --------------------------------
# Load scRNA-seq datasets

def loadZebrafishData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["stage.nice"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    # -----
    cell_set_meta = pd.read_csv("{}/cell_groups_meta.csv".format(data_dir), header=0, index_col=0)
    meta_data = pd.concat([meta_data, cell_set_meta.loc[meta_data.index, :]], axis=1)
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadDrosophilaData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/subsample_meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadWOTData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-meta_data.csv".format(data_dir, split_type), header=0, index_col=0)
    cell_idx = np.where(~np.isnan(meta_data["day"].values))[0] # remove cells with nan labels
    cnt_data = cnt_data.iloc[cell_idx, :]
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadVeresData(pt_path, split_type):
    # data = torch.load(pt_path, weights_only=False)
    data = torch.load(f'{pt_path}/{split_type}_fate_train.pt', weights_only=False)
    X = torch.cat(data['x'], dim=0).numpy()
    obs_list = []
    for t, celltype_series in zip(data['y'], data['celltype']):
        obs = pd.DataFrame({
            "CellWeek": t,
            "Assigned_cluster": celltype_series.values
        }, index=celltype_series.index)
        obs_list.append(obs)
    obs_all = pd.concat(obs_list, axis=0)
    cell_stage = obs_all["CellWeek"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.full(len(cell_stage), np.nan)
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    obs_all["tp"] = cell_tp
    obs_all["veres_cluster"] = obs_all["Assigned_cluster"]
    ann_data = scanpy.AnnData(X=X, obs=obs_all)
    return ann_data



def loadSCData(data_name, split_type, data_dir=None):
    '''
    Main function to load scRNA-seq dataset and pre-process it.
    '''
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        ann_data = loadZebrafishData(data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].apply(lambda x: "NAN" if pd.isna(x) else x).values
    elif data_name == "drosophila":
        ann_data = loadDrosophilaData(data_dir, split_type)
        print("Pre-processing...")
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = processed_data.obs.seurat_clusters.values
    elif data_name == "wot":
        ann_data = loadWOTData(data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = None
    elif data_name == "veres":
        ann_data = loadVeresData(data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = ann_data
        cell_types = processed_data.obs["veres_cluster"].values
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps


def tpSplitInd(data_name, split_type):
    '''
    Get the training/testing timepoint split for each dataset.
    '''
    if data_name == "zebrafish":
        if split_type == "two_forecasting": # forecasting task
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
            test_tps = [10, 11]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11]
            test_tps = [4, 6, 8]
        elif split_type == "remove_recovery": # random recovery task
            train_tps = [0, 1, 3, 5, 7, 9]
            test_tps = [2, 4, 6, 8, 10, 11]
        elif split_type == "all_times": # for perturbation prediction
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "drosophila":
        if split_type == "three_forecasting": # forecasting task
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7]
            test_tps = [8, 9, 10]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10]
            test_tps = [4, 6, 8]
        elif split_type == "remove_recovery":
            train_tps = [0, 1, 3, 5, 7]
            test_tps = [2, 4, 6, 8, 9, 10]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "wot":
        unique_days = np.arange(19)
        if split_type == "three_forecasting": # forecasting task
            train_tps = unique_days[:16].tolist()
            test_tps = unique_days[16:].tolist()
        elif split_type == "three_interpolation":
            train_tps = unique_days.tolist()
            test_tps = [train_tps[5], train_tps[10], train_tps[15]]
            train_tps.remove(unique_days[5])
            train_tps.remove(unique_days[10])
            train_tps.remove(unique_days[15])
        elif split_type == "remove_recovery": # random recovery task
            train_tps = unique_days.tolist()
            test_idx = [5, 7, 9, 11, 15, 16, 17, 18]
            test_tps = [train_tps[t] for t in test_idx]
            for t in test_idx:
                train_tps.remove(unique_days[t])
        else:
            raise ValueError("Unknown split type {}!".format(split_type))

    elif data_name == "veres":
        if split_type == "remove_recovery": # random recovery task
            train_tps = [0, 1, 3, 5]
            test_tps = [2, 4, 6, 7]
        elif split_type == "two_forecasting": # forecasting task
            train_tps = [0, 1, 2, 3, 4, 5]
            test_tps = [6, 7]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 3, 5, 7]
            test_tps = [2, 4, 6]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    else:
        raise ValueError("Unknown data name.")
    return train_tps, test_tps


def splitBySpec(traj_data, train_tps, test_tps):
    '''
    Split timepoints into training and testing sets.
    '''
    train_data = [traj_data[t] for t in train_tps]
    test_data = [traj_data[t] for t in test_tps]
    return train_data, test_data



def preprocess(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage

    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )

    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data

