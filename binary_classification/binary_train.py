"""Binary burst classifier training: 单卡 / 单机多卡 DDP / CPU 通用。

训练流程：
  - EMA + timm cosine + warmup 调度
  - 每 epoch 在验证集上对 EMA 和 raw 模型都跑评估（loss / acc / P / R / F1）
  - 按 EMA 的指定指标（``--best-metric``，默认 F1）挑 best checkpoint
  - 完整 checkpoint + ``--resume`` 续训；``--finetune <ckpt>`` 仅加载权重做新一轮初始化

启动方式：
  * 单卡：``python binary_train.py``
  * DDP：``torchrun --nproc_per_node=N binary_train.py``
"""

import argparse
import json
import os
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

from binary_data import BurstDataset, get_train_val
from binary_model import build_binary_model, random_resize


# ---------------------------------------------------------------------------
# 1. 通用辅助：CLI bool 解析 / 随机种子 / 分布式 / 设备
# ---------------------------------------------------------------------------

def parse_bool(value):
    """argparse 用：把 ``"1"/"true"/"yes"`` 解析为 True；其余对应 False；无效报错。"""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def setup_seed(seed):
    """同时设置 numpy + torch（CPU + 所有 CUDA 设备）的随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args):
    """读取 torchrun 注入的环境变量；未启动 DDP 时返回 (0, 0, 1)，主流程退化为单卡。"""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    kwargs = {"backend": backend, "timeout": timedelta(minutes=30)}
    if backend == "nccl" and torch.cuda.is_available():
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


def is_main(rank):
    return rank == 0


def setup_device(args, local_rank, world_size):
    """DDP 用 local_rank；单卡尊重 --device；显式 cpu 或无 CUDA 时回退到 CPU。"""
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if world_size > 1:
        return torch.device(f"cuda:{local_rank}")
    if args.device:
        return torch.device(f"cuda:{args.device}")
    return torch.device("cuda")


def default_log_dir(model_type, model_name, fix_size):
    """根据模型 / 训练规模给一个有提示性的默认 log dir。"""
    prefix = "logs_spp" if model_type == "SPPConvNeXt" else "logs_convnext"
    suffix = "fix" if fix_size else "ran"
    return f"./{prefix}_{model_name}_{suffix}/"


# ---------------------------------------------------------------------------
# 2. 数据：train / val (DDP 分片) + eval (rank 0 独占)
# ---------------------------------------------------------------------------

def build_loaders(args, rank, world_size):
    train_df, val_df = get_train_val(args.data_path, train_ratio=args.train_ratio, seed=args.seed)
    train_dataset = BurstDataset(train_df, val=False)
    val_dataset = BurstDataset(val_df, val=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        drop_last=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler, **loader_kw,
    )
    # rank 0 独占的非分布式 eval loader（用于跑 EMA 模型的完整评估）
    val_eval_loader = None
    if is_main(rank):
        val_eval_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
    return train_loader, train_sampler, val_loader, val_eval_loader


# ---------------------------------------------------------------------------
# 3. 指标累加：loss / acc / 正类的 P / R / F1
#    用一个 6 维 tensor 在 GPU 上累加，结束时一次 AllReduce 即可拿到全局值。
# ---------------------------------------------------------------------------

# stats 张量的分量含义（label 1 = 正类 / true burst）
_LOSS, _CORRECT, _TP, _FP, _FN, _TOTAL = range(6)


def _new_stats(device):
    return torch.zeros(6, dtype=torch.float64, device=device)


def _accumulate(stats, loss, outputs, labels):
    """把当前 batch 的 loss / 正确数 / TP / FP / FN / 样本数累加到 stats。"""
    pred = outputs.argmax(dim=1)
    pos_pred, pos_gt = pred == 1, labels == 1
    bs = labels.size(0)
    stats[_LOSS] += loss.detach().double() * bs
    stats[_CORRECT] += (pred == labels).sum()
    stats[_TP] += (pos_pred & pos_gt).sum()
    stats[_FP] += (pos_pred & ~pos_gt).sum()
    stats[_FN] += (~pos_pred & pos_gt).sum()
    stats[_TOTAL] += bs


def _finalize_stats(stats, world_size):
    """对 stats 做 AllReduce → 算出 loss / acc / P / R / F1 五个标量。"""
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total = max(stats[_TOTAL].item(), 1.0)
    tp, fp, fn = stats[_TP].item(), stats[_FP].item(), stats[_FN].item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "loss": stats[_LOSS].item() / total,
        "acc": 100.0 * stats[_CORRECT].item() / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
    }


def _fmt(m):
    return (f"loss={m['loss']:.4f}  acc={m['acc']:.2f}%  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")


# ---------------------------------------------------------------------------
# 4. 单 epoch 训练 / 验证循环
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled, train_mode, rank,
              world_size, args, scheduler=None, global_step=0, ema=None):
    """跑一个 epoch；返回（指标 dict, 更新后的 global_step）。"""
    model.train() if train_mode else model.eval()
    tag = "Train" if train_mode else "Val"
    stats = _new_stats(device)
    pbar = tqdm(loader, dynamic_ncols=True, ascii=True, disable=not is_main(rank), desc=tag)

    for inputs, labels in pbar:
        # 多尺度训练：每个 batch 随机选 128~512 的边长（先于送到设备）
        if train_mode and not args.fix_size:
            inputs = random_resize(inputs)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    global_step += 1
                    scheduler.step_update(global_step)

        _accumulate(stats, loss, outputs, labels)
        if is_main(rank):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return _finalize_stats(stats, world_size), global_step


@torch.no_grad()
def eval_model(model, loader, criterion, device, amp_enabled):
    """单进程评估（用于 EMA 模型，rank 0 独占）：返回 loss/acc/P/R/F1。"""
    model.eval()
    stats = _new_stats(device)
    for inputs, labels in tqdm(loader, dynamic_ncols=True, ascii=True, desc="EMA eval"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        _accumulate(stats, loss, outputs, labels)
    return _finalize_stats(stats, world_size=1)


# ---------------------------------------------------------------------------
# 5. checkpoint
# ---------------------------------------------------------------------------

def _state(model):
    """从（可能被 DDP 包裹的）模型上拿 state_dict。"""
    return (model.module if hasattr(model, "module") else model).state_dict()


def _load_weights_only(path, model, device):
    """仅加载权重（``--finetune`` 用，不读优化器 / 调度器）；兼容 state_dict / 完整 ckpt dict。"""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    base = model.module if hasattr(model, "module") else model
    base.load_state_dict(state, strict=True)


def _save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, global_step, best):
    """完整 checkpoint：足够 ``--resume`` 继续训练。"""
    torch.save({
        "epoch": epoch + 1,
        "global_step": global_step,
        "model_state_dict": _state(model),
        "ema_state_dict": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best": best,                                          # {metric_name: best_value}，多指标兼容
    }, path)


def _resume_from(path, model, ema, optimizer, scheduler, scaler, device):
    """从 :func:`_save_checkpoint` 写出的 dict 续训；返回 (start_epoch, global_step, best)。"""
    ckpt = torch.load(path, map_location=device)
    base = model.module if hasattr(model, "module") else model
    base.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    if "ema_state_dict" in ckpt:
        ema.ema.load_state_dict(ckpt["ema_state_dict"])
        ema.updates = int(ckpt.get("ema_updates", 0))
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), ckpt.get("best", {})


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------

# --best-metric → (在 metrics dict 中的键名, 是否越大越好)
_BEST_METRICS = {
    "f1":       ("f1", True),
    "recall":   ("recall", True),
    "acc":      ("acc", True),
    "val_loss": ("loss", False),
}


def main(args):
    setup_seed(args.seed)
    rank, local_rank, world_size = setup_distributed(args)
    device = setup_device(args, local_rank, world_size)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # batch size 默认值按是否 DDP / 是否 fix_size 自适应；用户传 >0 时尊重 CLI
    if args.batch_size <= 0:
        args.batch_size = 16 if world_size > 1 else (64 if args.fix_size else 32)

    log_dir = args.log_dir or default_log_dir(args.model_type, args.model_name, args.fix_size)
    if is_main(rank):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log dir: {log_dir}  |  batch_size={args.batch_size}")
    if world_size > 1:
        sync_barrier(local_rank)

    # ---- 数据 ----
    train_loader, train_sampler, val_loader, val_eval_loader = build_loaders(args, rank, world_size)

    # ---- 模型 ----
    model = build_binary_model(
        model_type=args.model_type, model_name=args.model_name,
        num_classes=args.num_classes, pretrained=args.pretrained, dropout=args.dropout,
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if args.finetune:
        _load_weights_only(args.finetune, model, device)
        if is_main(rank):
            print(f"Finetune init from {args.finetune}")

    # ModelEMA 的有效 decay 在前 ~tau 步内从 0 爬升到 ema_decay；tau 越小影子权重跟得越快
    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau)

    # ---- 优化器 / 调度器 ----
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs * steps_per_epoch,
        lr_min=args.lr_min,
        warmup_t=args.warmup_epochs * steps_per_epoch,
        warmup_lr_init=args.warmup_lr,
        t_in_epochs=False,
        cycle_limit=1,
    )
    scheduler.step_update(0)                                # 让 LR 立刻进入 warmup 起点
    amp_enabled = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # ---- 续训 ----
    start_epoch, global_step, best = 0, 0, {}
    if args.resume:
        start_epoch, global_step, best = _resume_from(args.resume, model, ema, optimizer, scheduler, scaler, device)
        if is_main(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
    best_key, best_higher = _BEST_METRICS[args.best_metric]
    best_value = best.get(args.best_metric, (-float("inf") if best_higher else float("inf")))

    # ---- 训练循环 ----
    logs = []
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_lr = optimizer.param_groups[0]["lr"]
        if is_main(rank):
            print(f"\n{'='*80}\n  Epoch {epoch + 1}/{args.epochs}  |  LR {epoch_lr:.6e}\n{'='*80}")

        train_m, global_step = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled,
            True, rank, world_size, args, scheduler=scheduler, global_step=global_step, ema=ema,
        )
        val_m, global_step = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, amp_enabled,
            False, rank, world_size, args, scheduler=None, global_step=global_step,
        )

        if not is_main(rank):
            sync_barrier(local_rank)
            continue

        # ---- 评估 + 日志（只在 rank 0）----
        ema_m = eval_model(ema.ema, val_eval_loader, criterion, device, amp_enabled)
        print(f"\n  {'─'*76}")
        print(f"  Train       {_fmt(train_m)}")
        print(f"  Val  [RAW]  {_fmt(val_m)}")
        print(f"  Val  [EMA]  {_fmt(ema_m)}")

        logs.append({
            "epoch": epoch + 1, "lr": epoch_lr,
            **{f"{k}_train": train_m[k] for k in train_m},
            **{f"{k}_val": val_m[k] for k in val_m},
            **{f"{k}_ema": ema_m[k] for k in ema_m},
        })
        with open(os.path.join(log_dir, "logs.json"), "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        # ---- 保存 ----
        epoch_name = (f"Epoch{epoch + 1:03d}_Tloss{train_m['loss']:.3f}"
                      f"_Vloss{val_m['loss']:.3f}_VF1{val_m['f1']:.3f}.pth")
        torch.save(_state(model), os.path.join(log_dir, epoch_name))
        torch.save(ema.ema.state_dict(), os.path.join(log_dir, epoch_name.replace(".pth", "_ema.pth")))

        # 按 EMA 验证指标挑 best，使 best_model_ema.pth 与对应指标的最优 epoch 对齐
        cur = ema_m[best_key]
        improved = (cur > best_value) if best_higher else (cur < best_value)
        if improved:
            best_value = cur
            best[args.best_metric] = best_value
            torch.save(_state(model), os.path.join(log_dir, "best_model.pth"))
            torch.save(ema.ema.state_dict(), os.path.join(log_dir, "best_model_ema.pth"))
            print(f"  ★ New best (EMA {args.best_metric}={best_value:.4f}), saved best_model.pth / best_model_ema.pth")

        _save_checkpoint(os.path.join(log_dir, "last_checkpoint.pth"),
                         model, ema, optimizer, scheduler, scaler, epoch, global_step, best)
        sync_barrier(local_rank)

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--finetune", type=str, default="",
                        help="只加载该权重做初始化，优化器 / 调度器从头开始")

    # model
    parser.add_argument("--model-type", type=str, default="ConvNeXtNet", choices=["ConvNeXtNet", "SPPConvNeXt"])
    parser.add_argument("--model-name", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5, help="仅 SPPConvNeXt 用到")
    parser.add_argument("--fix-size", type=parse_bool, default=True,
                        help="True=固定 512² 训练；False=每个 batch 随机 128~512 边长（多尺度）")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="加载 ImageNet 预训练 backbone（默认开；--no-pretrained 关闭）")

    # hardware
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "nccl", "gloo"])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    # data / training
    parser.add_argument("--batch-size", type=int, default=0, help="<=0 时按 fix_size / 是否 DDP 自动选")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-tau", type=int, default=1000, help="EMA decay 的爬升步数；越小影子权重跟得越快")
    parser.add_argument("--best-metric", type=str, default="f1", choices=list(_BEST_METRICS))

    main(parser.parse_args())
