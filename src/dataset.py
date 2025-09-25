import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.utils import loadSCData, tpSplitInd, splitBySpec

torch.set_default_dtype(torch.float32)




class TrainDataset(Dataset):
    def __init__(self, X_list, t_list, batch_size, split_type):
        self.X_list = X_list
        self.t_list = torch.tensor(t_list, dtype=torch.float32).unsqueeze(-1)
        self.batch_size = batch_size
        self.cell_counts = np.array([x.shape[0] for x in X_list])
        self.total_cells = self.cell_counts.sum()
        self.split_type = split_type
        self.context_len = 4
    def __len__(self):
        return self.total_cells // self.batch_size
    def __getitem__(self, idx):
        time_idx = np.random.choice(range(1, len(self.X_list)))
        x_target = self.X_list[time_idx].to(torch.float32)
        t_target = self.t_list[time_idx].to(torch.float32)

        available_obs_t = self.t_list[:time_idx].squeeze(-1).tolist()
        selected_obs_t = available_obs_t[-self.context_len:] if len(available_obs_t) >= self.context_len else [available_obs_t[0]] * (self.context_len - len(available_obs_t)) + available_obs_t

        raw_delta_t = [selected_obs_t[i+1] - selected_obs_t[i] for i in range(len(selected_obs_t) - 1)]
        delta_t = [self.context_len] * (self.context_len - len(raw_delta_t) - 1) + [0.0] + raw_delta_t        
        delta_t = torch.tensor(delta_t, dtype=torch.float32)

        t_mask = torch.tensor(
            [0 if t == available_obs_t[0] and i < (self.context_len - len(available_obs_t)) else 1 for i, t in enumerate(selected_obs_t)],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1).expand(self.batch_size, self.context_len, 1)        
        x_obs = torch.cat([
            self.X_list[available_obs_t.index(t)][np.random.choice(self.X_list[available_obs_t.index(t)].shape[0], self.batch_size, replace=True)].to(torch.float32).unsqueeze(1)
            for t in selected_obs_t
        ], dim=1)
        
        x_target = x_target[np.random.choice(x_target.shape[0], self.batch_size, replace=True)].unsqueeze(1)
        
        t_obs = torch.tensor(selected_obs_t, dtype=torch.float32).unsqueeze(0).expand(self.batch_size, self.context_len).to(torch.float32)
        t_target = t_target.expand(self.batch_size, 1).to(torch.float32)
        
        return x_obs, x_target, t_obs, t_target, t_mask, delta_t


class TestDataset(Dataset):
    def __init__(self, X_train_list, X_test_list, t_train_list, t_test_list, batch_size):
        self.X_train_list = X_train_list
        self.X_test_list = X_test_list
        self.t_train_list = torch.tensor(t_train_list, dtype=torch.float32).unsqueeze(-1)
        self.t_test_list = torch.tensor(t_test_list, dtype=torch.float32).unsqueeze(-1)
        self.batch_size = batch_size
        self.num_timepoints = len(X_train_list)
        self.context_len = 4
        self.index_map = []
        for i, x in enumerate(X_test_list):
            n_batches = x.shape[0] // batch_size
            self.index_map.extend([(i, j) for j in range(n_batches)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        time_idx, _ = self.index_map[idx]
        t_target = self.t_test_list[time_idx]
        
        available_obs_t = self.t_train_list[self.t_train_list < t_target].squeeze(-1).tolist()
        selected_obs_t = available_obs_t[-self.context_len:] if len(available_obs_t) >= self.context_len else [available_obs_t[0]] * (self.context_len - len(available_obs_t)) + available_obs_t

        raw_delta_t = [selected_obs_t[i+1] - selected_obs_t[i] for i in range(len(selected_obs_t) - 1)]
        delta_t = [self.context_len] * (self.context_len - len(raw_delta_t) - 1) + [0.0] + raw_delta_t        
        delta_t = torch.tensor(delta_t, dtype=torch.float32)
        
        t_mask = torch.tensor([0 if t == available_obs_t[0] and i < (self.context_len - len(available_obs_t)) else 1 for i, t in enumerate(selected_obs_t)], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(self.batch_size, self.context_len, 1)
        
        x_train = torch.cat([
            self.X_train_list[available_obs_t.index(t)][np.random.choice(self.X_train_list[available_obs_t.index(t)].shape[0], self.batch_size, replace=True)].unsqueeze(1).to(torch.float32)
            for t in selected_obs_t
        ], dim=1)
        
        t_train = torch.tensor(selected_obs_t, dtype=torch.float32).unsqueeze(0).expand(self.batch_size, self.context_len)
        t_target = t_target.expand(self.batch_size, 1)
        
        return x_train, t_train, t_target, t_mask, delta_t


class GeneExpressionDataset:
    def __init__(self, data_dir, data_name, split_type, batch_size):
        self.data_dir = data_dir
        self.data_name = data_name
        self.split_type = split_type
        self.batch_size = batch_size
        self.ann_data, self.cell_tps, _, self.gene_dim, self.n_tps = self.load_data()
        self.train_tps, self.test_tps = tpSplitInd(self.data_name, self.split_type)
        self.all_tps = sorted(set(self.train_tps + self.test_tps))
        self.traj_data = self.get_traj_data()
        self.test_dict, self.train_dataloader, self.test_dataloader = self.prepare_dataloader()

    def load_data(self):
        return loadSCData(self.data_name, self.split_type, self.data_dir)

    def prepare_dataloader(self):
        train_data, test_data = splitBySpec(self.traj_data, self.train_tps, self.test_tps)
        test_dict = dict(zip(self.test_tps, test_data))

        train_dataset = TrainDataset(train_data, self.train_tps, self.batch_size, self.split_type)
        test_dataset = TestDataset(train_data, test_data, self.train_tps, self.test_tps, self.batch_size)

        return (
            test_dict,
            DataLoader(train_dataset, batch_size=1, shuffle=True),
            DataLoader(test_dataset, batch_size=1, shuffle=False),
        )

    def get_traj_data(self):
        return [
            torch.tensor(self.ann_data.X[np.where(self.cell_tps == t)[0], :], dtype=torch.float32)
            for t in range(1, self.n_tps + 1)
        ]

