"""
VAE 潜变量提取与可视化 (UMAP/t-SNE)

Usage:
    python extract_latents.py \
        --h5_path ./data.h5 \
        --model_path ./outputs/best.pth \
        --encoder convnext \
        --method umap
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import VariationalRadioAutoencoder
from dataset_h5 import H5RadioDataset

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Use `pip install umap-learn`")

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("Warning: sklearn not installed for t-SNE")


def extract_latents(model, loader, device):
    """提取所有样本的潜变量"""
    model.eval()

    all_latents = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents", ascii=True):
            data = batch.to(device, non_blocking=True)

            # 提取潜变量 (全局平均池化)
            z = model.encode(data, pool='gap')  # [B, latent_ch]
            all_latents.append(z.cpu().numpy())

            # 计算异常分数
            score = model.get_anomaly_score(data, beta=0.1)
            all_scores.append(score.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)  # [N, latent_ch]
    scores = np.concatenate(all_scores, axis=0)    # [N]

    return latents, scores


def reduce_dimensions(latents, method='umap', n_components=2, random_state=42):
    """降维到 2D"""
    print(f"Reducing dimensions with {method.upper()}...")

    if method == 'umap':
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=random_state,
        )
    elif method == 'tsne':
        if not HAS_TSNE:
            raise ImportError("sklearn not installed")
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    embedding = reducer.fit_transform(latents)
    return embedding


def visualize(embedding, scores, output_path=None, title="Latent Space"):
    """可视化降维结果"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 按分数着色
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=scores,
        cmap='viridis',
        s=5,
        alpha=0.7,
    )

    plt.colorbar(scatter, ax=ax, label='Anomaly Score')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)

    # 标记高分异常点
    threshold = np.percentile(scores, 95)
    anomaly_mask = scores > threshold
    if anomaly_mask.sum() > 0:
        ax.scatter(
            embedding[anomaly_mask, 0],
            embedding[anomaly_mask, 1],
            c='red',
            s=20,
            marker='x',
            label=f'Top 5% anomalies (n={anomaly_mask.sum()})',
        )
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 加载数据 ==========
    print(f"\n[Step 1] Loading data: {args.h5_path}")
    dataset = H5RadioDataset(args.h5_path, return_meta=False)
    print(f"Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ========== 加载模型 ==========
    print(f"\n[Step 2] Loading VAE model: {args.model_path}")

    model = VariationalRadioAutoencoder(
        pretrained=False,
        latent_ch=args.latent_dim,
        encoder_name=args.encoder,
    )

    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    print(f"Model: VAE + {args.encoder.upper()}, Latent: {args.latent_dim}")

    # ========== 提取潜变量 ==========
    print("\n[Step 3] Extracting latent vectors...")
    latents, scores = extract_latents(model, loader, device)
    print(f"Latents shape: {latents.shape}")
    print(f"Scores - Mean: {scores.mean():.6f}, Std: {scores.std():.6f}")

    # 保存潜变量
    np.savez(
        output_dir / "latents.npz",
        latents=latents,
        scores=scores,
    )
    print(f"Saved: {output_dir / 'latents.npz'}")

    # ========== 降维 ==========
    print(f"\n[Step 4] Dimensionality reduction ({args.method})...")
    embedding = reduce_dimensions(latents, method=args.method)

    # ========== 可视化 ==========
    print("\n[Step 5] Visualizing...")
    visualize(
        embedding,
        scores,
        output_path=output_dir / f"latent_space_{args.method}.png",
        title=f"VAE Latent Space ({args.method.upper()})",
    )

    # 保存降维结果
    np.savez(
        output_dir / f"embedding_{args.method}.npz",
        embedding=embedding,
        scores=scores,
    )

    print("\nDone!")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='VAE Latent Space Visualization')

    # Input
    p.add_argument('--h5_path', type=str, required=True, help='Path to H5 file')

    # Model
    p.add_argument('--model_path', type=str, required=True, help='Path to VAE model weights')
    p.add_argument('--encoder', type=str, default='convnext', choices=['convnext', 'resnet18'])
    p.add_argument('--latent_dim', type=int, default=128, help='VAE latent channels')

    # Inference
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)

    # Visualization
    p.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'])
    p.add_argument('--output_dir', type=str, default='./latent_analysis')

    sys.exit(main(p.parse_args()))
