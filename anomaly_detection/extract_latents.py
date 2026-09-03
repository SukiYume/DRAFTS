# fmt: off

"""Extract VAE latent means and create a UMAP or t-SNE visualization."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset_h5 import H5RadioDataset
from models import VariationalRadioAutoencoder, compute_anomaly_scores
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent


def extract_latents(model, loader, device, beta):
    """Return pooled latent means and anomaly scores in loader order."""
    model.eval()
    latent_batches = []
    score_batches  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents", ascii=True):
            data                       = batch.to(device, non_blocking=True)
            reconstruction, mu, logvar = model(data)
            latent_batches.append(mu.mean(dim=(2, 3)).cpu().numpy())
            scores, _, _ = compute_anomaly_scores(
                reconstruction,
                data,
                mu,
                logvar,
                beta,
            )
            score_batches.append(scores.cpu().numpy())

    return np.concatenate(latent_batches), np.concatenate(score_batches)


def reduce_dimensions(latents, method="umap", random_state=42):
    """Reduce latent vectors to two dimensions."""
    sample_count = latents.shape[0]
    if sample_count < 3:
        raise ValueError(
            "At least three samples are required for dimensionality reduction"
        )

    print(f"Reducing dimensions with {method.upper()}...")
    if method == "umap":
        try:
            import umap
        except ImportError as error:
            raise ImportError("Install umap-learn to use UMAP") from error
        reducer = umap.UMAP(
            n_components = 2,
            n_neighbors  = min(15, sample_count - 1),
            min_dist     = 0.1,
            metric       = "euclidean",
            random_state = random_state,
            init         = "random",
        )
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as error:
            raise ImportError("Install scikit-learn to use t-SNE") from error
        reducer = TSNE(
            n_components = 2,
            perplexity   = min(30, sample_count - 1),
            random_state = random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(latents)


def visualize(embedding, scores, output_path, title="Latent space"):
    """Save a score-colored two-dimensional latent-space plot."""
    figure, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c     = scores,
        cmap  = "viridis",
        s     = 5,
        alpha = 0.7,
    )
    figure.colorbar(scatter, ax=axis, label="Anomaly score")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.set_title(title)

    threshold    = np.percentile(scores, 95)
    anomaly_mask = scores > threshold
    if anomaly_mask.any():
        axis.scatter(
            embedding[anomaly_mask, 0],
            embedding[anomaly_mask, 1],
            color  = "red",
            s      = 20,
            marker = "x",
            label  = f"Top 5% (n={anomaly_mask.sum()})",
        )
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main(args):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if min(args.batch_size, args.latent_dim) <= 0:
        raise ValueError("batch_size and latent_dim must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.kl_weight < 0:
        raise ValueError("kl_weight cannot be negative")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    dataset = H5RadioDataset(args.h5_path)
    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = False,
        num_workers        = args.num_workers,
        pin_memory         = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
    print(f"Samples: {len(dataset)}")

    model = VariationalRadioAutoencoder(
        pretrained   = False,
        latent_ch    = args.latent_dim,
        encoder_name = args.encoder,
    )
    checkpoint = torch.load(
        args.model_path,
        map_location = device,
        weights_only = True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("model_config", {})
        expected = {
            "model": "vae",
            "encoder": args.encoder,
            "latent_dim": args.latent_dim,
        }
        mismatches = {
            key: (config[key], value)
            for key, value in expected.items()
            if key in config and config[key] != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint configuration mismatch: {mismatches}")
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()

    latents, scores = extract_latents(
        model,
        loader,
        device,
        beta=args.kl_weight,
    )
    print(f"Latents: {latents.shape}")
    print(f"Scores: mean={scores.mean():.6f}, std={scores.std():.6f}")
    np.savez(output_dir / "latents.npz", latents=latents, scores=scores)

    embedding = reduce_dimensions(
        latents,
        method       = args.method,
        random_state = args.seed,
    )
    visualize(
        embedding,
        scores,
        output_dir / f"latent_space_{args.method}.png",
        title=f"VAE latent space ({args.method.upper()})",
    )
    np.savez(
        output_dir / f"embedding_{args.method}.npz",
        embedding = embedding,
        scores    = scores,
    )
    print(f"Done: {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--encoder",
        default = "convnext",
        choices = ["convnext", "resnet18"],
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(BASE_DIR / "latent_analysis"))
    sys.exit(main(parser.parse_args()))

# fmt: on
