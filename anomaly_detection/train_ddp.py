"""
DDP Training for Radio Telescope Anomaly Detection

Usage:
    torchrun --nproc_per_node=8 train_ddp_v2.py \
        --h5_path ./data.h5 --model ae --epochs 50 --augment

    torchrun --nproc_per_node=8 train_ddp_v2.py \
        --h5_path ./data.h5 --model vae --epochs 50 --augment \
        --kl_weight 0.1 --kl_warmup_epochs 10
"""

import os
os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

import h5py
import numpy as np

from models import (
    RadioAutoencoder,
    VariationalRadioAutoencoder,
    compute_ae_loss,
    compute_vae_loss,
)
from dataset_h5 import H5RadioDataset, H5PairDataset, DataAugmentation


# =============================================================================
# Distributed Utils
# =============================================================================

def setup_distributed():
    """Initialize DDP."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = 'gloo' # "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    else:
        rank, local_rank, world_size = 0, 0, 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Data Utils
# =============================================================================

def build_indices(h5_path, down_times=None, val_ratio=0.1, seed=42):
    """Build train/val indices from HDF5."""
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
        if down_times is None:
            indices = np.arange(n, dtype=np.int64)
        else:
            down = np.asarray(f["down_time"][:]).reshape(-1)
            indices = np.where(np.isin(down, down_times))[0].astype(np.int64)
            print(f"[Filter] down_times {down_times}: {len(indices)}/{n}")

    if len(indices) == 0:
        raise ValueError("No samples after filtering")

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(len(indices) * val_ratio)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    print(f"[Split] Train: {len(train_idx)}, Val: {len(val_idx)}")
    return train_idx, val_idx


def get_kl_beta(step, warmup_steps, beta_max):
    """KL annealing schedule."""
    if warmup_steps <= 0:
        return beta_max
    return beta_max * min(1.0, step / warmup_steps)

# =============================================================================
# Main
# =============================================================================

def main(args):
    setup_seed(args.seed)
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # TF32 for Ampere+
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if is_main(rank):
        print(f"\n{'='*60}\nDDP Training - {world_size} GPUs\n{'='*60}\n")

    # Output dir
    if is_main(rank):
        output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)

    if world_size > 1:
        dist.barrier()

    # ==================== Data ====================
    use_vae = args.model == "vae"
    use_pred = args.pred

    if use_pred:
        # 预测模式：使用配对数据集
        train_transform = DataAugmentation() if args.augment else None
        full_ds = H5PairDataset(args.h5_path, down_time_list=args.down_times, transform=train_transform)

        # 简单的 train/val 划分
        n_total = len(full_ds)
        n_val = int(n_total * args.val_ratio)
        indices = np.random.RandomState(args.seed).permutation(n_total)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(
            H5PairDataset(args.h5_path, down_time_list=args.down_times, transform=None),
            val_idx
        ) if n_val > 0 else None
    else:
        # AE/VAE 重建模式
        train_idx, val_idx = build_indices(args.h5_path, args.down_times, args.val_ratio, args.seed)
        train_transform = DataAugmentation() if args.augment else None
        train_ds = H5RadioDataset(args.h5_path, indices=train_idx, transform=train_transform)
        val_ds = H5RadioDataset(args.h5_path, indices=val_idx) if len(val_idx) > 0 else None

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) if val_ds and world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
    ) if val_ds else None

    if is_main(rank):
        mode_str = "PRED-" + args.model.upper() if use_pred else args.model.upper()
        print(f"Mode: {mode_str}")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}")
        print(f"Batch: {args.batch_size} x {world_size} = {args.batch_size * world_size}")

    # ==================== Model ====================
    if use_vae:
        model = VariationalRadioAutoencoder(
            pretrained=not args.no_pretrained,
            latent_ch=args.latent_dim,
            encoder_name=args.encoder
        )
    else:  # ae
        model = RadioAutoencoder(
            pretrained=not args.no_pretrained,
            encoder_name=args.encoder
        )

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    base_model = model.module if world_size > 1 else model

    if is_main(rank):
        pretrained_str = "ImageNet pretrained" if not args.no_pretrained else "random init"
        print(f"Model: {args.model.upper()} + {args.encoder.upper()}, Params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Encoder: {pretrained_str}\n")

    # ==================== Optimizer ====================
    encoder_params = list(base_model.encoder.parameters())
    decoder_params = list(base_model.decoder.parameters())
    if use_vae:
        decoder_params += list(base_model.to_mu.parameters())
        decoder_params += list(base_model.to_logvar.parameters())
        decoder_params += list(base_model.from_z.parameters())

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=0.05)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    kl_warmup_steps = args.kl_warmup_epochs * steps_per_epoch if use_vae else 0

    scheduler = CosineLRScheduler(
        optimizer, t_initial=total_steps, lr_min=1e-6,
        warmup_t=warmup_steps, warmup_lr_init=1e-6, t_in_epochs=False,
    )
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # ==================== Resume ====================
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val = ckpt.get("best_val_loss", float("inf"))
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}")

    logs = []

    # ==================== Training ====================
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if is_main(rank):
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_dec = optimizer.param_groups[1]["lr"]
            print(f"\nEpoch {epoch+1}/{args.epochs} | LR: enc={lr_enc:.2e}, dec={lr_dec:.2e}")

        # ----- Train -----
        model.train()
        stats = torch.zeros(4, dtype=torch.float64, device=device)  # loss, recon, kl, count
        pbar = tqdm(train_loader, desc="Train", disable=not is_main(rank), ascii=True)

        for batch in pbar:
            # pred 模式: batch = (x_prev, x_curr)
            # ae/vae 模式: batch = data
            if use_pred:
                x_prev, x_curr = batch
                x_prev = x_prev.to(device, non_blocking=True)
                x_curr = x_curr.to(device, non_blocking=True)
                bs = x_prev.size(0)
            else:
                data = batch.to(device, non_blocking=True)
                bs = data.size(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                if use_pred:
                    # 预测模式: 用 x_prev 预测 x_curr
                    if use_vae:
                        beta = get_kl_beta(global_step, kl_warmup_steps, args.kl_weight)
                        recon, mu, logvar = model(x_prev)
                        loss, recon_loss, kl_loss = compute_vae_loss(recon, x_curr, mu, logvar, beta)
                    else:
                        recon = model(x_prev)
                        loss, recon_loss, kl_loss = compute_ae_loss(recon, x_curr)
                else:
                    # 重建模式
                    if use_vae:
                        beta = get_kl_beta(global_step, kl_warmup_steps, args.kl_weight)
                        recon, mu, logvar = model(data)
                        loss, recon_loss, kl_loss = compute_vae_loss(recon, data, mu, logvar, beta)
                    else:
                        recon = model(data)
                        loss, recon_loss, kl_loss = compute_ae_loss(recon, data)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            scheduler.step_update(global_step)

            stats[0] += loss.detach().double() * bs
            stats[1] += recon_loss.detach().double() * bs
            stats[2] += kl_loss.detach().double() * bs if use_vae else 0
            stats[3] += bs

            pbar.set_postfix(loss=f"{loss.detach():.4f}")

        # ----- Validate -----
        val_stats = torch.zeros(4, dtype=torch.float64, device=device)
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val", disable=not is_main(rank), ascii=True):
                    if use_pred:
                        x_prev, x_curr = batch
                        x_prev = x_prev.to(device, non_blocking=True)
                        x_curr = x_curr.to(device, non_blocking=True)
                        bs = x_prev.size(0)
                    else:
                        data = batch.to(device, non_blocking=True)
                        bs = data.size(0)

                    with autocast("cuda", enabled=torch.cuda.is_available()):
                        if use_pred:
                            if use_vae:
                                recon, mu, logvar = model(x_prev)
                                loss, recon_loss, kl_loss = compute_vae_loss(recon, x_curr, mu, logvar, args.kl_weight)
                            else:
                                recon = model(x_prev)
                                loss, recon_loss, kl_loss = compute_ae_loss(recon, x_curr)
                        else:
                            if use_vae:
                                recon, mu, logvar = model(data)
                                loss, recon_loss, kl_loss = compute_vae_loss(recon, data, mu, logvar, args.kl_weight)
                            else:
                                recon = model(data)
                                loss, recon_loss, kl_loss = compute_ae_loss(recon, data)

                    val_stats[0] += loss.detach().double() * bs
                    val_stats[1] += recon_loss.detach().double() * bs
                    val_stats[2] += kl_loss.detach().double() * bs if use_vae else 0
                    val_stats[3] += bs

        # ----- Aggregate -----
        if world_size > 1:
            dist.all_reduce(stats)
            dist.all_reduce(val_stats)

        train_loss = (stats[0] / stats[3]).item()
        train_recon = (stats[1] / stats[3]).item()
        val_loss = (val_stats[0] / val_stats[3]).item() if val_stats[3] > 0 else train_loss
        val_recon = (val_stats[1] / val_stats[3]).item() if val_stats[3] > 0 else train_recon

        if is_main(rank):
            print(f"Train: loss={train_loss:.4f}, recon={train_recon:.4f} | Val: loss={val_loss:.4f}, recon={val_recon:.4f}")

            state = base_model.state_dict()

            # 1. 保存每个 epoch 的模型（文件名包含详细指标）
            epoch_name = f"Epoch{epoch+1:03d}_Tloss{train_loss:.4f}_Vloss{val_loss:.4f}.pth"
            torch.save(state, output_dir / epoch_name)
            print(f"  Saved: {epoch_name}")

            # 2. 保存最低 val loss 模型
            if val_loss < best_val:
                best_val = val_loss
                torch.save(state, output_dir / "best_loss_model.pth")
                print("  -> Best loss model updated!")

            # 3. 保存完整的 last checkpoint（用于断点续训）
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val,
            }
            torch.save(checkpoint, output_dir / "last_checkpoint.pth")
            print()

        logs.append([train_loss, val_loss])
        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        np.save(output_dir / "logs.npy", np.array(logs))
        print(f"\nDone! Output: {output_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", type=str, required=True)
    p.add_argument("--down_times", type=int, nargs="+", default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--model", type=str, default="ae", choices=["ae", "vae"])
    p.add_argument("--pred", action="store_true", help="Use temporal prediction mode (f(x_t-1) = x_t)")
    p.add_argument("--encoder", type=str, default="convnext", choices=["convnext", "resnet18"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder_lr_mult", type=float, default=0.3)
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--kl_warmup_epochs", type=int, default=10)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true")
    main(p.parse_args())
