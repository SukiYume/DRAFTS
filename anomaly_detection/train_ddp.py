# fmt: off

"""Train AE/VAE radio anomaly models on one GPU or with torchrun DDP."""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from dataset_h5 import DataAugmentation, H5PairDataset, H5RadioDataset
from models import (
    RadioAutoencoder,
    VariationalRadioAutoencoder,
    compute_ae_loss,
    compute_vae_loss,
)
from timm.scheduler import CosineLRScheduler
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent


def setup_distributed():
    """Initialize torchrun state and return rank, local rank, and world size."""
    if "RANK" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = (
        "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    )
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return rank, local_rank, world_size


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _split_by_source(source_names, val_ratio, seed):
    """Keep each FITS source wholly in train or validation when possible."""
    source_names   = np.asarray(source_names)
    sample_indices = np.arange(source_names.size, dtype=np.int64)
    if val_ratio == 0:
        return sample_indices, np.empty(0, dtype=np.int64)

    rng            = np.random.RandomState(seed)
    unique_sources = np.unique(source_names)
    if unique_sources.size > 1:
        rng.shuffle(unique_sources)
        validation_source_count = min(
            unique_sources.size - 1,
            max(1, round(unique_sources.size * val_ratio)),
        )
        validation_mask = np.isin(
            source_names, unique_sources[:validation_source_count]
        )
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    rng.shuffle(sample_indices)
    validation_count = int(sample_indices.size * val_ratio)
    return sample_indices[validation_count:], sample_indices[:validation_count]


def build_indices(h5_path, down_times=None, val_ratio=0.1, seed=42):
    """Filter by downsampling factor and split without cross-source leakage."""
    with h5py.File(h5_path, "r") as h5:
        sample_count = h5["images"].shape[0]
        names_raw    = np.asarray(h5["names"][:]).reshape(-1)
        if down_times is None:
            indices = np.arange(sample_count, dtype=np.int64)
        else:
            factors = np.asarray(h5["down_time"][:]).reshape(-1)
            indices = np.flatnonzero(np.isin(factors, down_times))

    if indices.size == 0:
        raise ValueError("No samples after down_time filtering")
    names = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in names_raw
        ]
    )
    train_positions, validation_positions = _split_by_source(
        names[indices], val_ratio, seed
    )
    return indices[train_positions], indices[validation_positions]


def get_kl_beta(step, warmup_steps, beta_max):
    """Linearly anneal beta from zero to its configured maximum."""
    if warmup_steps <= 0:
        return beta_max
    return beta_max * min(1.0, step / warmup_steps)


def main(args):
    if not os.path.isfile(args.h5_path):
        raise FileNotFoundError(f"HDF5 not found: {args.h5_path}")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if min(args.epochs, args.batch_size, args.latent_dim) <= 0:
        raise ValueError("epochs, batch_size, and latent_dim must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.lr <= 0 or args.encoder_lr_mult <= 0:
        raise ValueError("learning rates must be positive")
    if args.kl_weight < 0 or min(args.warmup_epochs, args.kl_warmup_epochs) < 0:
        raise ValueError("KL weight and warmup epochs cannot be negative")

    rank, local_rank, world_size = setup_distributed()
    setup_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    main_process = rank == 0
    if main_process:
        print(f"\n{'=' * 60}\nTraining on {world_size} process(es)\n{'=' * 60}\n")
        output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = Path(args.output_dir)
    if world_size > 1:
        dist.barrier()

    use_vae         = args.model == "vae"
    train_transform = DataAugmentation() if args.augment else None

    if args.pred:
        train_source = H5PairDataset(
            args.h5_path,
            down_time_list = args.down_times,
            transform      = train_transform,
        )
        train_indices, validation_indices = _split_by_source(
            train_source.source_names,
            args.val_ratio,
            args.seed,
        )
        train_dataset = torch.utils.data.Subset(train_source, train_indices)
        if validation_indices.size:
            validation_source = H5PairDataset(
                args.h5_path,
                down_time_list = args.down_times,
                transform      = None,
            )
            validation_dataset = torch.utils.data.Subset(
                validation_source, validation_indices
            )
        else:
            validation_dataset = None
    else:
        train_indices, validation_indices = build_indices(
            args.h5_path,
            args.down_times,
            args.val_ratio,
            args.seed,
        )
        train_dataset = H5RadioDataset(
            args.h5_path,
            indices   = train_indices,
            transform = train_transform,
        )
        validation_dataset = (
            H5RadioDataset(args.h5_path, indices=validation_indices)
            if validation_indices.size
            else None
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas = world_size,
            rank         = rank,
            shuffle      = True,
        )
        if world_size > 1
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas = world_size,
            rank         = rank,
            shuffle      = False,
        )
        if validation_dataset is not None and world_size > 1
        else None
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle   = train_sampler is None,
        sampler   = train_sampler,
        drop_last = False,
        **loader_options,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            shuffle   = False,
            sampler   = validation_sampler,
            drop_last = False,
            **loader_options,
        )
        if validation_dataset is not None
        else None
    )

    if use_vae:
        model = VariationalRadioAutoencoder(
            pretrained   = not args.no_pretrained,
            latent_ch    = args.latent_dim,
            encoder_name = args.encoder,
        )
    else:
        model = RadioAutoencoder(
            pretrained   = not args.no_pretrained,
            encoder_name = args.encoder,
        )
    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    base_model = model.module if world_size > 1 else model

    if main_process:
        mode           = f"PRED-{args.model.upper()}" if args.pred else args.model.upper()
        initialization = "random" if args.no_pretrained else "ImageNet pretrained"
        print(f"Mode: {mode} + {args.encoder.upper()} ({initialization})")
        print(
            f"Train: {len(train_dataset)}, "
            f"Val: {len(validation_dataset) if validation_dataset is not None else 0}"
        )
        print(
            f"Batch: {args.batch_size} x {world_size} = {args.batch_size * world_size}"
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"Parameters: {parameter_count:,}\n")

    encoder_parameters = list(base_model.encoder.parameters())
    decoder_parameters = list(base_model.decoder.parameters())
    if use_vae:
        decoder_parameters.extend(base_model.to_mu.parameters())
        decoder_parameters.extend(base_model.to_logvar.parameters())
        decoder_parameters.extend(base_model.from_z.parameters())

    optimizer = optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.lr * args.encoder_lr_mult,
            },
            {"params": decoder_parameters, "lr": args.lr},
        ],
        weight_decay=0.05,
    )
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = min(args.warmup_epochs * steps_per_epoch, total_steps)
    kl_warmup_steps = args.kl_warmup_epochs * steps_per_epoch if use_vae else 0
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial      = total_steps,
        lr_min         = 1e-6,
        warmup_t       = warmup_steps,
        warmup_lr_init = 1e-6,
        t_in_epochs    = False,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch          = 0
    global_step          = 0
    best_validation_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        config     = checkpoint.get("model_config", {})
        expected_config = {
            "model": args.model,
            "encoder": args.encoder,
            "pred": args.pred,
        }
        if use_vae:
            expected_config["latent_dim"] = args.latent_dim
        mismatches = {
            key: (config[key], value)
            for key, value in expected_config.items()
            if key in config and config[key] != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint configuration mismatch: {mismatches}")
        base_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch          = int(checkpoint["epoch"])
        global_step          = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint.get("best_val_loss", float("inf")))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch}; "
                f"requested epochs={args.epochs}"
            )
        if main_process:
            print(f"Resumed from epoch {start_epoch}")

    logs = []
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if main_process:
            print(
                f"\nEpoch {epoch + 1}/{args.epochs} | "
                f"LR: enc={optimizer.param_groups[0]['lr']:.2e}, "
                f"dec={optimizer.param_groups[1]['lr']:.2e}"
            )

        model.train()
        train_stats = torch.zeros(4, dtype=torch.float64, device=device)
        progress = tqdm(
            train_loader,
            desc    = "Train",
            disable = not main_process,
            ascii   = True,
        )
        for batch in progress:
            inputs, targets = batch if args.pred else (batch, batch)
            inputs          = inputs.to(device, non_blocking=True)
            targets         = targets.to(device, non_blocking=True)
            batch_size      = inputs.size(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                if use_vae:
                    beta                       = get_kl_beta(global_step, kl_warmup_steps, args.kl_weight)
                    reconstruction, mu, logvar = model(inputs)
                    loss, recon_loss, kl_loss = compute_vae_loss(
                        reconstruction,
                        targets,
                        mu,
                        logvar,
                        beta,
                    )
                else:
                    reconstruction            = model(inputs)
                    loss, recon_loss, kl_loss = compute_ae_loss(reconstruction, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            scheduler.step_update(global_step)

            train_stats[0] += loss.detach().double() * batch_size
            train_stats[1] += recon_loss.detach().double() * batch_size
            train_stats[2] += kl_loss.detach().double() * batch_size
            train_stats[3] += batch_size
            progress.set_postfix(loss=f"{loss.detach():.4f}")

        validation_stats = torch.zeros(4, dtype=torch.float64, device=device)
        if validation_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    validation_loader,
                    desc    = "Val",
                    disable = not main_process,
                    ascii   = True,
                ):
                    inputs, targets = batch if args.pred else (batch, batch)
                    inputs          = inputs.to(device, non_blocking=True)
                    targets         = targets.to(device, non_blocking=True)
                    batch_size      = inputs.size(0)

                    with autocast("cuda", enabled=device.type == "cuda"):
                        if use_vae:
                            reconstruction, mu, logvar = model(inputs)
                            loss, recon_loss, kl_loss = compute_vae_loss(
                                reconstruction,
                                targets,
                                mu,
                                logvar,
                                args.kl_weight,
                            )
                        else:
                            reconstruction = model(inputs)
                            loss, recon_loss, kl_loss = compute_ae_loss(
                                reconstruction, targets
                            )

                    validation_stats[0] += loss.detach().double() * batch_size
                    validation_stats[1] += recon_loss.detach().double() * batch_size
                    validation_stats[2] += kl_loss.detach().double() * batch_size
                    validation_stats[3] += batch_size

        if world_size > 1:
            dist.all_reduce(train_stats)
            dist.all_reduce(validation_stats)

        train_loss  = (train_stats[0] / train_stats[3]).item()
        train_recon = (train_stats[1] / train_stats[3]).item()
        train_kl    = (train_stats[2] / train_stats[3]).item()
        if validation_stats[3] > 0:
            validation_loss  = (validation_stats[0] / validation_stats[3]).item()
            validation_recon = (validation_stats[1] / validation_stats[3]).item()
            validation_kl    = (validation_stats[2] / validation_stats[3]).item()
        else:
            validation_loss  = train_loss
            validation_recon = train_recon
            validation_kl    = train_kl

        if main_process:
            summary = (
                f"Train: loss={train_loss:.4f}, recon={train_recon:.4f} | "
                f"Val: loss={validation_loss:.4f}, recon={validation_recon:.4f}"
            )
            if use_vae:
                summary += f" | KL: train={train_kl:.4f}, val={validation_kl:.4f}"
            print(summary)
            state = base_model.state_dict()
            epoch_name = (
                f"Epoch{epoch + 1:03d}_Tloss{train_loss:.4f}_"
                f"Vloss{validation_loss:.4f}.pth"
            )
            torch.save(state, output_dir / epoch_name)
            print(f"  Saved: {epoch_name}")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(state, output_dir / "best_loss_model.pth")
                print("  Best loss model updated")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_validation_loss,
                    "model_config": {
                        "model": args.model,
                        "encoder": args.encoder,
                        "latent_dim": args.latent_dim,
                        "pred": args.pred,
                    },
                },
                output_dir / "last_checkpoint.pth",
            )

        logs.append([train_loss, validation_loss])
        if world_size > 1:
            dist.barrier()

    if main_process:
        np.save(output_dir / "logs.npy", np.asarray(logs))
        print(f"\nDone! Output: {output_dir}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--down_times", type=int, nargs="+", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--model", default="ae", choices=["ae", "vae"])
    parser.add_argument(
        "--pred",
        action = "store_true",
        help   = "Predict each slice from the preceding slice",
    )
    parser.add_argument(
        "--encoder",
        default = "convnext",
        choices = ["convnext", "resnet18"],
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr_mult", type=float, default=0.3)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--output_dir", default=str(BASE_DIR / "outputs"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    main(parser.parse_args())

# fmt: on
