"""CenterNet detector training: 单卡 / 单机多卡 DDP / CPU 通用。

训练流程：
  - EMA + timm cosine + warmup 调度
  - close-mosaic 收尾（最后若干 epoch 关掉 mosaic 拼图）
  - 每 epoch 在验证集上对 EMA 和原模型都跑一次评估
  - 用召回优先 score 挑 best checkpoint：
    ``0.50*recall + 0.20*precision + 0.30*center_quality``
  - 完整 checkpoint + ``--resume`` 续训

启动方式：
  * 单卡：``python centernet_train.py``
  * DDP：``torchrun --nproc_per_node=N centernet_train.py``
  分布式专用代码都以 ``world_size > 1`` 为前置条件，未启动 DDP 时自动退化为单卡路径。
"""

import argparse
import json
import os
import random
from collections import deque
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from timm.scheduler import CosineLRScheduler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ultralytics.utils.torch_utils import ModelEMA

from centernet_data import H5CenterNetDataset, centernet_collate_fn, get_train_val
from centernet_eval import evaluate_metrics
from centernet_model import build_centernet_model, compute_loss


# ---------------------------------------------------------------------------
# 1. 通用辅助：随机种子 / 分布式 / 设备
# ---------------------------------------------------------------------------

def setup_seed(seed):
    """统一设置 Python random、numpy、torch（CPU + 所有 CUDA 设备）的随机种子。

    train DataLoader 的 shuffle 用全局 torch RNG（已被此处 :func:`torch.manual_seed`
    播种）；DataLoader worker 内的随机性另由 :func:`_make_worker_init_fn` 单独散播。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """读取 torchrun 注入的环境变量；未启动 DDP 时返回 (0, 0, 1)，主流程退化为单卡。"""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    kwargs = {"backend": backend, "timeout": timedelta(minutes=30)}
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        kwargs["device_id"] = local_rank
    dist.init_process_group(**kwargs)
    return rank, local_rank, world_size


def sync_barrier(local_rank):
    """跨 rank 同步；非 DDP 场景是 no-op。"""
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _pick_device(args, local_rank, world_size):
    """DDP 用 local_rank；单卡尊重 --device；显式 cpu 或无 CUDA 时回退到 CPU。"""
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if world_size > 1:
        return torch.device(f"cuda:{local_rank}")
    if args.device:
        return torch.device(f"cuda:{args.device}")
    return torch.device("cuda")


def _make_worker_init_fn(rank, base_seed):
    """显式分散 DataLoader worker 的 numpy/random/torch 随机序列。"""
    def init(worker_id):
        seed = (int(base_seed) * 10007 + int(rank) * 1009 + int(worker_id)) % (2 ** 31 - 1)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return init


# ---------------------------------------------------------------------------
# 2. 数据：train (DDP 分片) / val (DDP 分片) / eval (rank 0 独占)
# ---------------------------------------------------------------------------

def _build_loaders(args, rank, world_size):
    train_df, val_df = get_train_val(args.data_path, train_ratio=args.train_ratio, seed=args.seed)
    common = dict(imgsz=args.imgsz, down_ratio=args.down_ratio, max_objs=args.max_objs,
                  center_radius=args.center_radius)
    train_data = H5CenterNetDataset(train_df, val=False, **common)
    val_data = H5CenterNetDataset(val_df, val=True, **common)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    worker_init_fn = _make_worker_init_fn(rank, args.seed) if args.num_workers > 0 else None
    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=centernet_collate_fn,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        drop_last=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler, **loader_kw,
    )

    # rank 0 独占的非分布式 eval loader：让中心距离匹配/F1/score 在完整验证集上跑一次；
    # 只迭代一次/epoch，不需要常驻 worker
    val_eval_loader = None
    if rank == 0:
        val_eval_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            **{**loader_kw, "persistent_workers": False},
        )
    return train_data, train_loader, train_sampler, val_loader, val_eval_loader


# ---------------------------------------------------------------------------
# 3. 单 epoch 训练 / 验证循环
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, scaler, amp_enabled, train_mode, rank, args,
              scheduler=None, global_step=0, ema=None):
    """跑一个 epoch；返回该 rank 上的累加 (loss/hm/offset/count) 与更新后的 global_step。"""
    model.train() if train_mode else model.eval()
    tag = "Train" if train_mode else "Valid"
    pbar = tqdm(loader, dynamic_ncols=True, ascii=True, disable=(rank != 0), desc=tag)

    sums = {k: torch.zeros(1, dtype=torch.float64, device=device) for k in ("loss", "hm", "offset", "count")}

    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bs = batch["img"].shape[0]

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with autocast("cuda", enabled=amp_enabled):
                outputs = model(batch["img"])
                loss, parts = compute_loss(
                    outputs, batch, args.hm_weight, args.offset_weight,
                    args.hm_pos_weight, args.hm_neg_weight,
                )

            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    global_step += 1
                    scheduler.step_update(global_step)

        # compute_loss 返回的是「按本 batch 正样本数归一」的标量；
        # 累加时按 batch size 加权，得到整集的样本加权平均（仅用于日志）。
        sums["loss"] += loss.detach().double() * bs
        sums["hm"] += parts["hm"].double() * bs
        sums["offset"] += parts["offset"].double() * bs
        sums["count"] += bs

        if rank == 0:
            n = max(sums["count"].item(), 1.0)
            pbar.set_description(
                f"{tag} [loss={sums['loss'].item() / n:.4f}]"
                f"[hm={sums['hm'].item() / n:.4f}]"
                f"[off={sums['offset'].item() / n:.4f}]"
            )

    return sums, global_step


def _reduce_and_avg(sums, world_size):
    """对 sums 做 AllReduce 后按 count 求平均；返回不含 count 的 dict。"""
    if world_size > 1:
        for v in sums.values():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
    n = max(sums["count"].item(), 1.0)
    return {k: (sums[k].item() / n) for k in ("loss", "hm", "offset")}


# ---------------------------------------------------------------------------
# 4. checkpoint
# ---------------------------------------------------------------------------

def _unwrap(model):
    """剥掉可能的 DDP 包装，返回原始模型。"""
    return model.module if hasattr(model, "module") else model


def _save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, global_step, best_score, logs):
    """完整 checkpoint：足够 ``--resume`` 继续训练。"""
    torch.save({
        "epoch": epoch + 1,
        "global_step": global_step,
        "best_score": best_score,
        "model_state_dict": _unwrap(model).state_dict(),
        "ema_state_dict": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "logs": logs,
    }, path)


def _resume_from(path, model, ema, optimizer, scheduler, scaler, device, steps_per_epoch):
    """从 :func:`_save_checkpoint` 写出的 dict 续训；返回 (start_epoch, global_step, best_score, logs)。"""
    ckpt = torch.load(path, map_location=device)
    base = _unwrap(model)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        base.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", start_epoch * steps_per_epoch))
        best_score = float(ckpt.get("best_score", -float("inf")))
        logs = ckpt.get("logs", [])
        if "ema_state_dict" in ckpt:
            ema.ema.load_state_dict(ckpt["ema_state_dict"])
            ema.updates = int(ckpt.get("ema_updates", 0))
    else:
        # 兼容只存了 state_dict 的旧文件
        base.load_state_dict(ckpt, strict=True)
        start_epoch, global_step, best_score, logs = 0, 0, -float("inf"), []

    return start_epoch, global_step, best_score, logs


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def main(args):
    setup_seed(args.seed)
    rank, local_rank, world_size = setup_distributed()
    device = _pick_device(args, local_rank, world_size)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_name = args.backbone.replace("convnext_", "conv_")
    log_dir = args.log_dir or f"./logs/logs_centernet_{log_name}/"
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    if world_size > 1:
        sync_barrier(local_rank)

    # ---- 数据 ----
    train_data, train_loader, train_sampler, val_loader, val_eval_loader = _build_loaders(args, rank, world_size)

    # ---- 模型 ----
    # DDP + 预训练时多个 rank 会各自调用 torch.hub 下载 torchvision 权重；torch.hub 用原子重命名，并发安全。
    # 若担心重复下载，先用单卡跑一次把缓存填好即可。
    model = build_centernet_model(
        backbone=args.backbone, pretrained=args.pretrained,
        down_ratio=args.down_ratio, head_ch=args.head_ch,
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ModelEMA 的有效 decay 在前 ~tau 步内从 0 爬升到 ema_decay；tau 越小，影子权重跟原模型越紧
    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau)

    # ---- 优化器 / 调度器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    cosine_steps = (args.epochs - args.warmup_epochs) * steps_per_epoch
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cosine_steps,
        lr_min=args.lr_min,
        warmup_t=warmup_steps,
        warmup_lr_init=args.warmup_lr,
        t_in_epochs=False,
        warmup_prefix=True,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # ---- 落盘训练配置（rank 0）：args + DDP/batch 派生信息，方便事后追溯 ----
    if rank == 0:
        run_info = {
            **vars(args),
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * args.epochs,
        }
        with open(os.path.join(log_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=4, ensure_ascii=False, default=str)

    # ---- 续训 ----
    start_epoch, global_step, best_score, logs = 0, 0, -float("inf"), []
    if args.resume:
        start_epoch, global_step, best_score, logs = _resume_from(
            args.resume, model, ema, optimizer, scheduler, scaler, device, steps_per_epoch,
        )

    close_mosaic_epoch = args.epochs - args.close_mosaic_epochs

    # ---- SWA：rank 0 维护最近 N 个 EMA state_dict 的 CPU 副本，结束后求均值 ----
    swa_window = (
        deque(maxlen=args.swa_last_n)
        if rank == 0 and args.swa_last_n > 0 else None
    )

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练收尾：最后若干个 epoch 关掉 mosaic 拼图，让模型对真实单图分布做最后的对齐
        if epoch == close_mosaic_epoch and train_data.mosaic_prob > 0:
            train_data.mosaic_prob = 0.0
            if rank == 0:
                print(f"Mosaic disabled for last {args.close_mosaic_epochs} epochs")

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"  Epoch {epoch + 1}/{args.epochs}  |  LR {optimizer.param_groups[0]['lr']:.6e}")
            print(f"{'='*80}")

        train_sums, global_step = run_epoch(
            model, train_loader, optimizer, device, scaler, amp_enabled,
            True, rank, args, scheduler=scheduler, global_step=global_step, ema=ema,
        )
        val_sums, global_step = run_epoch(
            model, val_loader, optimizer, device, scaler, amp_enabled,
            False, rank, args, scheduler=None, global_step=global_step,
        )

        train_avg = _reduce_and_avg(train_sums, world_size)
        val_avg = _reduce_and_avg(val_sums, world_size)

        if rank != 0:
            sync_barrier(local_rank)
            continue

        # ---- 评估 + 日志（只在 rank 0）----
        print(f"\n  {'─'*76}")
        print(f"  Train Loss: {train_avg['loss']:.6f}  │  hm={train_avg['hm']:.4f}  off={train_avg['offset']:.4f}")
        print(f"  Valid Loss: {val_avg['loss']:.6f}  │  hm={val_avg['hm']:.4f}  off={val_avg['offset']:.4f}")

        eval_kw = dict(conf_thr=args.eval_conf, topk=args.topk, down_ratio=args.down_ratio, dist_thr=args.dist_thr)
        metrics = evaluate_metrics(ema.ema, val_eval_loader, device, **eval_kw)
        raw_model_eval = _unwrap(model)
        raw_model_eval.eval()
        metrics_raw = evaluate_metrics(raw_model_eval, val_eval_loader, device, **eval_kw)

        # EMA 通常更稳；按 EMA 的 score 挑 best，raw 仅打印做对照
        print(f"  {'─'*76}")
        for tag, m in (("EMA", metrics), ("RAW", metrics_raw)):
            print(
                f"  [{tag}] score={m['score']:.4f}  │  "
                f"F1={m['f1']:.4f} @ conf={m['f1_conf']:.3f}  │  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  │  "
                f"p90_dist={m['p90_center_dist']:.3f}px  CQ={m['center_quality']:.4f}  "
                f"(thr={m['dist_thr']:g})"
            )

        # logs.json 同时保留 EMA / RAW 两套指标
        metric_keys = (
            "score", "f1", "f1_conf", "precision", "recall",
            "center_quality", "mean_center_dist",
            "p50_center_dist", "p90_center_dist", "p95_center_dist",
        )
        logs.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"{k}_train": train_avg[k] for k in ("loss", "hm", "offset")},
            **{f"{k}_val": val_avg[k] for k in ("loss", "hm", "offset")},
            **{k: metrics[k] for k in metric_keys},
            **{f"{k}_raw": metrics_raw[k] for k in metric_keys},
            "dist_thr": metrics["dist_thr"],
        })
        with open(os.path.join(log_dir, "logs_centernet.json"), "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        # ---- 保存 ----
        epoch_name = f"Epoch{epoch + 1:03d}_TLoss{train_avg['loss']:.4f}_VLoss{val_avg['loss']:.4f}.pth"
        torch.save(_unwrap(model).state_dict(), os.path.join(log_dir, epoch_name))
        torch.save(ema.ema.state_dict(), os.path.join(log_dir, epoch_name.replace(".pth", "_ema.pth")))

        # 用 EMA 的 score 挑 best：
        # score = 0.50*recall + 0.20*precision + 0.30*max(0, 1 - p90_center_dist / dist_thr)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            print(f"  ★ New best (EMA score={best_score:.4f}), saving best_model.pth / best_model_ema.pth")
            torch.save(_unwrap(model).state_dict(), os.path.join(log_dir, "best_model.pth"))
            torch.save(ema.ema.state_dict(), os.path.join(log_dir, "best_model_ema.pth"))

        _save_checkpoint(
            os.path.join(log_dir, "last_checkpoint.pth"),
            model, ema, optimizer, scheduler, scaler, epoch, global_step, best_score, logs,
        )

        # SWA：记录该 epoch 的 EMA 权重（CPU 副本），最后 N 个用于平均
        if swa_window is not None:
            swa_window.append({k: v.detach().clone().cpu() for k, v in ema.ema.state_dict().items()})

        sync_barrier(local_rank)

    # ---- 训练结束后：把最后 N 个 EMA state_dict 平均，保存 swa_model_ema.pth ----
    if rank == 0 and swa_window is not None and len(swa_window) > 0:
        # BN running_mean / running_var 是激活分布的统计，每个 snapshot 都是基于自身那一刻
        # 权重统计出来的；权重一旦平均，分布漂移，再做线性平均的 stats 与平均后的权重不匹配。
        # PyTorch 官方 SWA (torch.optim.swa_utils.update_bn) 的做法是平均后用训练集重新刷一遍 BN，
        # 这里取一个更轻量但仍稳定的近似：BN buffer 全部从最后一个 snapshot 取。
        swa_state = {}
        last_sd = swa_window[-1]
        for k in swa_window[0].keys():
            tensors = [sd[k] for sd in swa_window]
            is_bn_buffer = (
                k.endswith(".running_mean")
                or k.endswith(".running_var")
                or k.endswith(".num_batches_tracked")
            )
            if is_bn_buffer or not tensors[0].is_floating_point():
                # BN stats / 整数 buffer：取最后一个 snapshot
                swa_state[k] = last_sd[k]
            else:
                stacked = torch.stack([t.float() for t in tensors])
                swa_state[k] = stacked.mean(dim=0).to(tensors[0].dtype)
        torch.save(swa_state, os.path.join(log_dir, "swa_model_ema.pth"))
        print(f"SWA: averaged last {len(swa_window)} EMA snapshots (weights mean, "
              f"BN stats from last) -> swa_model_ema.pth")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")

    # hardware
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度训练（仅 CUDA 生效；不加该参数则全精度）")

    # data
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--down-ratio", type=int, default=4)
    parser.add_argument("--max-objs", type=int, default=128)
    parser.add_argument("--center-radius", type=int, default=3,
                        help="heatmap 高斯核的*最小*半径；实际半径按目标尺寸自适应（gaussian_radius）。"
                             "调大可拓宽中心监督的有效区域，对压住推理 p90 偏差长尾通常有效")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)

    # model
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "convnext_tiny", "convnext_small"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="加载 ImageNet 预训练 backbone（默认开；--no-pretrained 关闭）")
    parser.add_argument("--head-ch", type=int, default=128)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-tau", type=int, default=1000, help="EMA decay 的爬升步数；越小影子权重跟得越快")
    parser.add_argument("--close-mosaic-epochs", type=int, default=10)
    parser.add_argument("--swa-last-n", type=int, default=10,
                        help="训练结束时把最后 N 个 epoch 的 EMA 权重平均，"
                             "保存为 swa_model_ema.pth；<=0 关闭")
    parser.add_argument("--hm-weight", type=float, default=1.0)
    parser.add_argument("--hm-pos-weight", type=float, default=1.0)
    parser.add_argument("--hm-neg-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=1.0)

    # eval (best checkpoint 用 EMA 的召回优先 score 挑)
    parser.add_argument("--eval-conf", type=float, default=0.01, help="评估时的置信度过滤阈值（用低阈值得到完整 PR 曲线）")
    parser.add_argument("--topk", type=int, default=100, help="每图保留的最大候选数")
    parser.add_argument("--dist-thr", type=float, default=8.0,
                        help="中心距离匹配阈值（像素）；同时用作 p90_center_dist 的归一化分母")

    main(parser.parse_args())
