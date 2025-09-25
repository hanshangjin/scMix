import os
os.environ["RWKV_FLOAT_MODE"] = "float32"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_T_MAX"] = "18"

import random
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from src.model import scMix
from src.layer import GPT
from src.dataset import GeneExpressionDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="remove_recovery", choices=["remove_recovery", "two_forecasting", "three_forecasting", "three_interpolation"], help="Training mode")
    parser.add_argument("--data_dir", type=str, default="/media/udata/time/new_data/data/single_cell/experimental/zebrafish_embryonic/new_processed", help="Directory containing dataset")
    # parser.add_argument("--data_dir", type=str, default="./new_data/data/single_cell/experimental/drosophila_embryonic/processed", help="Directory containing dataset")
    # parser.add_argument("--data_dir", type=str, default="./new_data/data/single_cell/experimental/Schiebinger2019/reduce_processed", help="Directory containing dataset")
    # parser.add_argument("--data_dir", type=str, default="./veres/processed", help="Directory containing dataset")
    parser.add_argument("--data_name", type=str, default="zebrafish", choices=["zebrafish", "drosophila", "wot", "veres"], help="Name of dataset")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--trend_coefficient", type=float, default=10.0, help="coefficient of trend regularization")
    parser.add_argument("--task_name", type=str, default="zebrafish_remove_recovery")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of RWKV layers")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--head_qk", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--n_embd", type=int, default=2048)
    parser.add_argument("--my_pos_emb", type=int, default=0)
    parser.add_argument("--pre_ffn", type=int, default=0)
    parser.add_argument("--grad_cp", type=int, default=0)
    parser.add_argument("--head_size_a", type=int, default=64)
    parser.add_argument("--head_size_divisor", type=int, default=8)
    parser.add_argument("--train_type", type=str, default="default")
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_3")
    parser.add_argument("--my_testing", default='x052', type=str)
    parser.add_argument("--model_type", default='RWKV', type=str)
    parser.add_argument("--dim_att", type=int, default=2048)
    parser.add_argument("--dim_ffn", type=int, default=2048)
    return parser.parse_args()


def train(args):
    dataset = GeneExpressionDataset(data_dir=args.data_dir, data_name=args.data_name, split_type=args.mode, batch_size=args.batch_size)
    train_dataloader = dataset.train_dataloader
    test_dataloader = dataset.test_dataloader
    test_dict = dataset.test_dict
    traj_data = dataset.traj_data
    gene_dim = dataset.gene_dim
    
    args.vocab_size = gene_dim
    args.gene_dim = gene_dim
    args.ctx_len = dataset.n_tps
    rwkv_base = GPT(args).to('cuda')
    model = scMix(base_model=rwkv_base, emb_dim=gene_dim, 
                            test_dict=test_dict, test_loader=test_dataloader, 
                            traj_data=traj_data, lr=args.lr, save_results=False, 
                            trend_coefficient=args.trend_coefficient, task_name=args.task_name).to('cuda')
    
    ws_name = os.path.basename(os.path.abspath(os.getcwd()))
    log_path = os.path.join("./ckpt/", ws_name)
    logger = CSVLogger(save_dir=log_path, name=f"{args.data_name}_{args.mode}_{args.max_epochs}")
    checkpoint_callback = ModelCheckpoint(save_last=False, save_top_k=0)
    trainer = pl.Trainer(max_epochs=args.max_epochs, devices=args.gpus, 
                        accelerator="gpu" if args.gpus > 0 else "cpu", 
                        log_every_n_steps=10, gradient_clip_val=1.0,
                        limit_val_batches=0, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    args = parse_args()
    train(args)
