#!/usr/bin/env python
"""
推理脚本：从 H5 文件或 FITS 文件进行异常检测

用法:
    # 从已有 H5 文件推理
    python scripts/run_inference.py --h5_path ./data.h5 --model_path ./best.pth

    # 从 FITS 文件推理（先转换再推理）
    python scripts/run_inference.py --fits_dir ./fits_data --model_path ./best.pth

    # 使用 VAE 模型
    python scripts/run_inference.py --h5_path ./data.h5 --model_path ./best.pth --model vae
"""

import os, sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()

from models import RadioAutoencoder, VariationalRadioAutoencoder, compute_ae_loss, compute_vae_loss
from dataset_h5 import H5RadioDataset, H5PairDataset
from prepare_data import build_h5_from_fits_paths


def run_inference(model, loader, device, use_vae=False, use_pred=False, beta=0.1):
    """运行推理，返回每个样本的异常分数"""
    model.eval()

    all_scores = []
    all_recon_losses = []
    all_kl_losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ascii=True):
            if use_pred:
                # 预测模式: batch = (x_prev, x_curr)
                x_prev, x_curr = batch
                x_prev = x_prev.to(device, non_blocking=True)
                x_curr = x_curr.to(device, non_blocking=True)
                target = x_curr

                if use_vae:
                    recon, mu, logvar = model(x_prev)
                    loss, recon_loss, kl_loss = compute_vae_loss(recon, x_curr, mu, logvar, beta)
                    all_kl_losses.extend([kl_loss.item()] * x_prev.size(0))
                else:
                    recon = model(x_prev)
                    loss, recon_loss, _ = compute_ae_loss(recon, x_curr)
            else:
                # 重建模式
                data = batch.to(device, non_blocking=True)
                target = data

                if use_vae:
                    recon, mu, logvar = model(data)
                    loss, recon_loss, kl_loss = compute_vae_loss(recon, data, mu, logvar, beta)
                    all_kl_losses.extend([kl_loss.item()] * data.size(0))
                else:
                    recon = model(data)
                    loss, recon_loss, _ = compute_ae_loss(recon, data)

            # 计算异常分数
            err_map = F.smooth_l1_loss(recon, target, reduction="none")
            err_flat = err_map.flatten(1)

            score_mean = err_flat.mean(dim=1)
            score_p99 = torch.quantile(err_flat, 0.99, dim=1)
            score = torch.maximum(score_mean, score_p99)

            all_scores.extend(score.detach().cpu().numpy())
            all_recon_losses.append(recon_loss.item())

    results = {
        'scores': np.array(all_scores),
        'recon_loss': np.mean(all_recon_losses),
    }

    if use_vae:
        results['kl_losses'] = np.array(all_kl_losses)

    return results


def compute_threshold(scores, method='3sigma'):
    """计算异常阈值"""
    if method == '3sigma':
        return scores.mean() + 3 * scores.std()
    elif method == 'iqr':
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        return q3 + 1.5 * iqr
    elif method == 'percentile':
        return np.percentile(scores, 95)
    else:
        return scores.mean() + 3 * scores.std()


def plot_results(scores, threshold, output_dir):
    """绘制分数分布和结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 分数分布
    ax = axes[0]
    ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Anomaly Score (SmoothL1 p99/max)')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.legend()

    # 排序分数
    ax = axes[1]
    sorted_scores = np.sort(scores)[::-1]
    ax.plot(sorted_scores, marker='.', markersize=2, linestyle='-', linewidth=0.5)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Sample Index (sorted)')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Sorted Scores')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'score_distribution.png'}")


def plot_top_anomalies(model, h5_path, scores, top_n, output_dir, device, use_vae=False):
    """可视化 Top N 异常样本（使用元数据作为标题）"""
    import h5py
    model.eval()

    top_indices = np.argsort(scores)[::-1][:top_n]

    # 直接从 H5 文件读取数据和元数据
    with h5py.File(h5_path, "r") as h5:
        for rank, idx in enumerate(top_indices):
            # 读取图像
            data_np = h5["images"][idx].astype(np.float32)
            data = torch.from_numpy(data_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            # 读取元数据
            name_raw = h5["names"][idx][0]
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
            name = Path(name).stem  # 只取文件名（不含路径和扩展名）
            down_time = int(h5["down_time"][idx][0])
            slice_idx = int(h5["slice_idx"][idx][0])

            with torch.no_grad():
                if use_vae:
                    recon, _, _ = model(data)
                else:
                    recon = model(data)

            orig = data[0, 0].cpu().numpy()
            recon_img = recon[0, 0].cpu().numpy()
            diff = np.abs(orig - recon_img)

            # 使用 percentile 确定 colorbar 范围
            vmin, vmax = np.percentile(orig, [1, 99])

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Original (转置：时间在 x 轴，频率在 y 轴)
            im0 = axes[0].imshow(orig.T, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax, origin='lower')
            axes[0].set_title('Original')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Frequency')
            plt.colorbar(im0, ax=axes[0])

            # Reconstructed
            im1 = axes[1].imshow(recon_img.T, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax, origin='lower')
            axes[1].set_title('Reconstructed')
            axes[1].set_xlabel('Time')
            plt.colorbar(im1, ax=axes[1])

            # Difference
            im2 = axes[2].imshow(diff.T, aspect='auto', cmap='mako', origin='lower')
            axes[2].set_title(f'Difference (Score: {scores[idx]:.4f})')
            axes[2].set_xlabel('Time')
            plt.colorbar(im2, ax=axes[2])

            # 使用有意义的标题：文件名、下采样率、slice 序号
            title = f"Rank {rank+1} | {name} | down_time={down_time} | slice={slice_idx}"
            fig.suptitle(title, fontsize=12)
            plt.tight_layout()

            # 文件名也更有意义
            save_name = f"anomaly_{rank+1:02d}_{name}_dt{down_time}_s{slice_idx}.png"
            plt.savefig(output_dir / save_name, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved: {output_dir}/ ({top_n} images)")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print("Radio Telescope Anomaly Detection - Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 准备数据 ==========
    if args.h5_path and os.path.exists(args.h5_path):
        print(f"\n[Step 1] Loading H5: {args.h5_path}")
        h5_path = args.h5_path
    elif args.fits_dir:
        print("\n[Step 1] Converting FITS to H5...")
        temp_h5 = output_dir / "temp_data.h5"
        h5_path, _, _ = build_h5_from_fits_paths(
            fits_paths=[args.fits_dir],
            out_h5_path=str(temp_h5),
            down_time_list=args.down_times,
            num_workers=args.num_workers,
        )
        print(f"Created: {h5_path}")
    else:
        print("Error: Please provide --h5_path or --fits_dir")
        return 1

    # 加载数据集
    use_vae = args.model == 'vae'
    use_pred = args.pred

    if use_pred:
        dataset = H5PairDataset(h5_path, down_time_list=args.down_times)
    else:
        dataset = H5RadioDataset(h5_path, return_meta=False)
    print(f"Samples: {len(dataset)}")

    # ========== 加载模型 ==========
    print(f"\n[Step 2] Loading model: {args.model_path}")

    if use_vae:
        model = VariationalRadioAutoencoder(
            pretrained=False,
            latent_ch=args.latent_dim,
            encoder_name=args.encoder
        )
    else:
        model = RadioAutoencoder(
            pretrained=False,
            encoder_name=args.encoder
        )

    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    mode_str = "PRED-" + args.model.upper() if use_pred else args.model.upper()
    print(f"Model: {mode_str} + {args.encoder.upper()}, Params: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 推理 ==========
    print("\n[Step 3] Running inference...")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results = run_inference(model, loader, device, use_vae, use_pred, args.kl_weight)
    scores = results['scores']

    # ========== 分析结果 ==========
    print("\n[Step 4] Analyzing results...")

    threshold = compute_threshold(scores, args.threshold_method)
    anomalies = scores > threshold
    n_anomalies = anomalies.sum()

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Score - Mean: {scores.mean():.6f}, Std: {scores.std():.6f}")
    print(f"Threshold ({args.threshold_method}): {threshold:.6f}")
    print(f"Anomalies: {n_anomalies} / {len(scores)} ({100*n_anomalies/len(scores):.1f}%)")

    # 保存结果
    np.savez(
        output_dir / 'results.npz',
        scores=scores,
        threshold=threshold,
        anomalies=anomalies,
    )
    print(f"\nSaved: {output_dir / 'results.npz'}")

    # 绘图
    plot_results(scores, threshold, output_dir)

    # 保存异常列表
    if n_anomalies > 0:
        with open(output_dir / 'anomalies.txt', 'w') as f:
            f.write(f"# Threshold: {threshold:.6f}\n")
            f.write(f"# Total anomalies: {n_anomalies}\n\n")
            for idx in np.argsort(scores)[::-1]:
                if anomalies[idx]:
                    f.write(f"{scores[idx]:.6f}\tsample_{idx}\n")
        print(f"Saved: {output_dir / 'anomalies.txt'}")

    # ========== 可视化 Top 异常 ==========
    if args.plot_anomalies:
        print("\n[Step 5] Generating anomaly visualizations...")
        anomaly_dir = output_dir / 'anomaly_images'
        anomaly_dir.mkdir(parents=True, exist_ok=True)
        plot_top_anomalies(model, h5_path, scores, args.top_n, anomaly_dir, device, use_vae)

    # Top 10 异常
    print("\nTop 10 Anomalies:")
    for i, idx in enumerate(np.argsort(scores)[::-1][:10]):
        status = "⚠️" if anomalies[idx] else ""
        print(f"  {i+1}. Score: {scores[idx]:.6f} - Sample {idx} {status}")

    print(f"\nDone! Results saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Radio Telescope Anomaly Detection Inference')

    # Input
    p.add_argument('--h5_path', type=str, default=None, help='Path to H5 file')
    p.add_argument('--fits_dir', type=str, default=None, help='Directory containing FITS files')
    p.add_argument('--down_times', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256],
                   help='Time downsampling factors for FITS conversion')

    # Model
    p.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    p.add_argument('--model', type=str, default='ae', choices=['ae', 'vae'])
    p.add_argument('--pred', action='store_true', help='Use temporal prediction mode')
    p.add_argument('--encoder', type=str, default='convnext', choices=['convnext', 'resnet18'])
    p.add_argument('--latent_dim', type=int, default=128, help='VAE latent channels')
    p.add_argument('--kl_weight', type=float, default=0.1, help='KL weight for VAE')

    # Inference
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--threshold_method', type=str, default='3sigma',
                   choices=['3sigma', 'iqr', 'percentile'])

    # Output
    p.add_argument('--output_dir', type=str, default='./inference_results')
    p.add_argument('--plot_anomalies', action='store_true', help='Generate anomaly visualizations')
    p.add_argument('--top_n', type=int, default=20, help='Number of top anomalies to visualize')

    sys.exit(main(p.parse_args()))
