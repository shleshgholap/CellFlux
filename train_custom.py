"""
Single-file training script for UNet on custom multiplex images (42x256x256)
with BBBC-style drug embeddings.

Requirements:
- PyTorch

Usage example:
python train_custom.py \
  --images_root /path/to/images \
  --index_csv /path/to/index.csv \
  --embeddings_csv /path/to/drug_embeddings.csv \
  --epochs 100 --batch_size 8 --lr 2e-4

Assumptions:
- The index CSV has at least columns: SAMPLE_KEY, CPD_NAME
- Images are stored as .npy or .pt tensors under images_root with filenames
  matching SAMPLE_KEY and shape [42, 256, 256]
- embeddings_csv is a table indexed by CPD_NAME with embedding dim 1024
"""

import os
import math
import time
import argparse
import random
import pandas as pd
import numpy as np
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from single_model import UNetModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiplexDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, images_root: str, emb_table: pd.DataFrame):
        super().__init__()
        self.df = index_df.reset_index(drop=True)
        self.images_root = images_root
        self.emb_table = emb_table
        assert "SAMPLE_KEY" in self.df.columns, "index CSV must include SAMPLE_KEY"
        assert "CPD_NAME" in self.df.columns, "index CSV must include CPD_NAME"

    def __len__(self):
        return len(self.df)

    def _load_image(self, key: str) -> torch.Tensor:
        # Tries .npy first then .pt
        npy_path = os.path.join(self.images_root, f"{key}.npy")
        pt_path = os.path.join(self.images_root, f"{key}.pt")
        if os.path.isfile(npy_path):
            arr = np.load(npy_path)
            x = torch.tensor(arr, dtype=torch.float32)
        elif os.path.isfile(pt_path):
            x = torch.load(pt_path)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
        else:
            raise FileNotFoundError(f"Missing image file for key {key} in {self.images_root}")
        assert x.ndim == 3 and x.shape[0] == 42 and x.shape[1] == 256 and x.shape[2] == 256, (
            f"Expected image shape [42,256,256], got {tuple(x.shape)} for key {key}")
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        key = row["SAMPLE_KEY"]
        drug = row["CPD_NAME"]
        x = self._load_image(key)  # [42,256,256]
        emb = self.emb_table.loc[drug].values.astype(np.float32)
        emb = torch.tensor(emb, dtype=torch.float32)
        return {"x": x, "cond": emb}


def make_dataloaders(index_csv: str, images_root: str, embeddings_csv: str, batch_size: int, num_workers: int, val_split: float = 0.1):
    df = pd.read_csv(index_csv)
    assert "SAMPLE_KEY" in df.columns and "CPD_NAME" in df.columns
    emb = pd.read_csv(embeddings_csv, index_col=0)
    # Ensure CPD_NAME alignment
    missing = set(df["CPD_NAME"]) - set(emb.index)
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} compounds, e.g. {list(missing)[:5]}")

    # Simple split
    perm = np.random.permutation(len(df))
    n_val = int(len(df) * val_split)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    tr_ds = MultiplexDataset(df.iloc[tr_idx], images_root, emb)
    val_ds = MultiplexDataset(df.iloc[val_idx], images_root, emb)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_dl, val_dl


# Simple DDPM-like noise schedule helpers
def get_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    train_loader, val_loader = make_dataloaders(
        args.index_csv, args.images_root, args.embeddings_csv, args.batch_size, args.num_workers, args.val_split
    )

    model = UNetModel(
        in_channels=42,
        out_channels=42,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attn_res),
        dropout=args.dropout,
        channel_mult=tuple(args.channel_mult),
        use_checkpoint=args.ckpt,
        condition_dim=args.condition_dim,
        use_scale_shift_norm=True,
        use_new_attention_order=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Diffusion setup
    T = args.timesteps
    betas = get_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(x0, t, noise):
        a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            cond = batch["cond"].to(device)
            B = x.shape[0]
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise)
            extra = {"concat_conditioning": cond}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(x_t, t, extra)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1
        train_loss = running / max(1, len(train_loader))

        # Validation (predict noise on a fixed t)
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                x = batch["x"].to(device)
                cond = batch["cond"].to(device)
                B = x.shape[0]
                t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(x)
                x_t = q_sample(x, t, noise)
                extra = {"concat_conditioning": cond}
                pred = model(x_t, t, extra)
                val_loss = F.mse_loss(pred, noise)
                val_running += val_loss.item()
            val_loss = val_running / max(1, len(val_loader))

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")

        if (epoch % args.save_every == 0) or is_best:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_val": best_val,
            }
            name = "best.pt" if is_best else f"epoch_{epoch}.pt"
            torch.save(ckpt, os.path.join(args.out_dir, name))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", type=str, required=True)
    p.add_argument("--index_csv", type=str, required=True)
    p.add_argument("--embeddings_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--model_channels", type=int, default=128)
    p.add_argument("--num_res_blocks", type=int, default=4)
    p.add_argument("--attn_res", type=int, nargs="+", default=[2])
    p.add_argument("--channel_mult", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--condition_dim", type=int, default=1024)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

