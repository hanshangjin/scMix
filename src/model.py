import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from geomloss import SamplesLoss
from src.evaluation import globalEvaluation


class scMix(pl.LightningModule):
    def __init__(self, base_model, emb_dim, test_dict, test_loader, traj_data, lr, save_results, trend_coefficient, task_name):
        super().__init__()
        self.gene_projector = nn.Linear(emb_dim, emb_dim)
        self.gene_norm = nn.LayerNorm(emb_dim)
        self.time_emb = nn.Linear(1, emb_dim)
        self.time_projector = nn.Linear(emb_dim, emb_dim)
        self.rwkv = base_model
        self.loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
        self.test_data = test_dict
        self.test_loader = test_loader
        self.traj_data = traj_data
        self.lr = lr
        self.test_outputs = []
        self.save_results = save_results
        self.trend_coefficient = trend_coefficient
        self.task_name = task_name


    def forward(self, x_gene, t_obs, t_target, delta_t):
        x_gene = torch.cat([x_gene, torch.zeros_like(x_gene[:, :1, :])], dim=1)

        x_gene = self.gene_projector(x_gene)
        x_gene = self.gene_norm(x_gene)

        delta_t = torch.cat([delta_t, torch.tensor([t_target[0].item() - t_obs[0, -1].item()], dtype=delta_t.dtype, device=delta_t.device)])

        t_obs = torch.cat([t_obs, t_target], dim=1)

        t_obs_emb = self.time_emb(t_obs.unsqueeze(-1))
        t_target_emb = self.time_emb(t_target.unsqueeze(-1))
        dist_t_emb = self.time_emb((t_target - t_obs).unsqueeze(-1))

        time_cond = t_obs_emb + t_target_emb + dist_t_emb
        x = x_gene + self.time_projector(time_cond)

        x = self.rwkv(x, delta_t)
        return x

    def training_step(self, batch, batch_idx):
        x_obs, x_target, t_obs, t_target, t_mask, delta_t = batch
        x_obs = x_obs.squeeze(0)
        x_target = x_target.squeeze(0)
        t_obs = t_obs.squeeze(0)
        t_target = t_target.squeeze(0)
        delta_t = delta_t.squeeze(0)

        x_pred = self.forward(x_obs, t_obs, t_target, delta_t)  # (B, T+1, D)

        loss = self.loss_fn(x_pred[:, -1, :], x_target.squeeze(1))
        loss_recon = self.loss_fn(x_pred[:, :-1, :].contiguous(), x_obs).mean()
        loss = loss + loss_recon

        dt = (t_obs - t_target).abs().clamp(min=1e-6)
        dx = ((x_obs - x_pred[:, -1, :].unsqueeze(1)) ** 2).mean(dim=2)
        trend_loss = (dx / (dt ** 2)).sum()

        total_loss = loss + self.trend_coefficient * trend_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("trend_loss", trend_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x_obs, t_obs, t_target, t_mask, delta_t = batch
        x_obs = x_obs.squeeze(0)
        t_obs = t_obs.squeeze(0)
        t_target = t_target.squeeze(0)
        delta_t = delta_t.squeeze(0)

        with torch.no_grad():
            x_pred = self.forward(x_obs, t_obs, t_target, delta_t)
            x_pred = x_pred[:, -1, :]

        return self.test_outputs.append({"x_pred": x_pred, "t_target": t_target})

    def on_test_epoch_end(self):
        t_targets = torch.cat([out["t_target"].to(torch.float32) for out in self.test_outputs], dim=0).cpu().numpy()
        all_preds = torch.cat([out["x_pred"].to(torch.float32) for out in self.test_outputs], dim=0).cpu().numpy()
        
        unique_times = np.unique(t_targets)
        print(f"test times: {unique_times}")

        true_results = dict()
        pred_results = dict()
        for time in unique_times:
            mask = t_targets == time
            mask = mask.reshape(-1)
            final_pred = all_preds[mask]
            final_target = self.test_data[time]

            final_metric = globalEvaluation(final_target, final_pred)

            print(f"Final Eval for Time {time} -> " f"OT={final_metric['ot']:.4f}")

            true_results[time] = final_target
            pred_results[time] = final_pred
            
        if self.save_results:
            all_times = list(range(len(self.traj_data)))
            test_times = set(unique_times)
            train_times = [t for t in all_times if t not in test_times]
            for t in train_times:
                true_results[int(t)] = self.traj_data[t]
            np.save(f"results/{self.task_name}_true.npy", true_results)
            np.save(f"results/{self.task_name}_pred.npy", pred_results)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
        

