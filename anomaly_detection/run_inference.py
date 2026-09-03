#!/usr/bin/env python
# fmt: off

"""Run AE/VAE anomaly detection on an HDF5 dataset or PSRFITS inputs."""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset_h5 import H5PairDataset, H5RadioDataset
from models import (
    RadioAutoencoder,
    VariationalRadioAutoencoder,
    compute_anomaly_scores,
)
from prepare_data import DEFAULT_DOWN_TIMES, build_h5_from_fits_paths
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent


def run_inference(model, loader, device, use_vae=False, use_pred=False, beta=0.1):
    """Return total, reconstruction, and KL scores in loader order."""
    model.eval()
    total_scores          = []
    reconstruction_scores = []
    kl_scores             = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ascii=True):
            inputs, targets = batch if use_pred else (batch, batch)
            inputs          = inputs.to(device, non_blocking=True)
            targets         = targets.to(device, non_blocking=True)

            if use_vae:
                reconstruction, mu, logvar = model(inputs)
                scores = compute_anomaly_scores(
                    reconstruction,
                    targets,
                    mu,
                    logvar,
                    beta,
                )
            else:
                reconstruction = model(inputs)
                scores         = compute_anomaly_scores(reconstruction, targets)

            total, reconstruction_score, kl = scores
            total_scores.append(total.cpu().numpy())
            reconstruction_scores.append(reconstruction_score.cpu().numpy())
            kl_scores.append(kl.cpu().numpy())

    return {
        "scores": np.concatenate(total_scores),
        "reconstruction_scores": np.concatenate(reconstruction_scores),
        "kl_scores": np.concatenate(kl_scores),
    }


def compute_threshold(scores, method="3sigma"):
    """Compute a threshold from a finite, non-empty score vector."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must be non-empty and finite")
    if method == "3sigma":
        return float(scores.mean() + 3 * scores.std())
    if method == "iqr":
        first_quartile, third_quartile = np.percentile(scores, [25, 75])
        return float(third_quartile + 1.5 * (third_quartile - first_quartile))
    if method == "percentile":
        return float(np.percentile(scores, 95))
    raise ValueError(f"Unknown threshold method: {method}")


def plot_results(scores, threshold, output_dir):
    """Save score-distribution and ranked-score plots."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(scores, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        threshold,
        color     = "red",
        linestyle = "--",
        linewidth = 2,
        label     = f"Threshold: {threshold:.4f}",
    )
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score distribution")
    axes[0].legend()

    sorted_scores = np.sort(scores)[::-1]
    axes[1].plot(sorted_scores, marker=".", markersize=2, linewidth=0.5)
    axes[1].axhline(
        threshold,
        color     = "red",
        linestyle = "--",
        linewidth = 2,
        label     = "Threshold",
    )
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Anomaly score")
    axes[1].set_title("Ranked scores")
    axes[1].legend()

    figure.tight_layout()
    output_path = output_dir / "score_distribution.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def plot_top_anomalies(
    model,
    h5_path,
    scores,
    input_indices,
    target_indices,
    names,
    down_times,
    slice_indices,
    top_n,
    output_dir,
    device,
    use_vae=False,
    use_pred=False,
):
    """Save model inputs, targets, reconstructions, and differences."""
    model.eval()
    ranked_positions = np.argsort(scores)[::-1][:top_n]

    with h5py.File(h5_path, "r") as h5:
        for rank, position in enumerate(ranked_positions, start=1):
            input_image = np.clip(
                h5["images"][input_indices[position]].astype(np.float32),
                0.0,
                1.0,
            )
            target_image = np.clip(
                h5["images"][target_indices[position]].astype(np.float32),
                0.0,
                1.0,
            )
            input_tensor = (
                torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                if use_vae:
                    reconstruction, _, _ = model(input_tensor)
                else:
                    reconstruction = model(input_tensor)
            reconstructed_image = reconstruction[0, 0].cpu().numpy()
            difference          = np.abs(target_image - reconstructed_image)

            panels = []
            if use_pred:
                panels.append((input_image, "Previous slice", False))
            panels.extend(
                [
                    (target_image, "Target" if use_pred else "Original", False),
                    (
                        reconstructed_image,
                        "Prediction" if use_pred else "Reconstruction",
                        False,
                    ),
                    (
                        difference,
                        f"Absolute difference\nscore={scores[position]:.4f}",
                        True,
                    ),
                ]
            )

            lower, upper = np.percentile(target_image, [1, 99])
            figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
            for axis, (image, title, is_difference) in zip(axes, panels, strict=True):
                image_options = {
                    "aspect": "auto",
                    "cmap": "viridis",
                    "origin": "lower",
                }
                if not is_difference and upper > lower:
                    image_options.update(vmin=lower, vmax=upper)
                rendered = axis.imshow(image.T, **image_options)
                axis.set_title(title)
                axis.set_xlabel("Time")
                if axis is axes[0]:
                    axis.set_ylabel("Frequency")
                figure.colorbar(rendered, ax=axis)

            figure.suptitle(
                f"Rank {rank} | {Path(names[position]).name} | "
                f"down_time={down_times[position]} | slice={slice_indices[position]}"
            )
            figure.tight_layout()
            stem      = Path(names[position]).stem or "sample"
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
            output_path = output_dir / (
                f"anomaly_{rank:02d}_{safe_stem}_dt{down_times[position]}_"
                f"s{slice_indices[position]}.png"
            )
            figure.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(figure)

    print(f"Saved: {output_dir} ({len(ranked_positions)} images)")


def main(args):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if min(args.batch_size, args.top_n, args.latent_dim) <= 0:
        raise ValueError("batch_size, top_n, and latent_dim must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.kl_weight < 0:
        raise ValueError("kl_weight cannot be negative")
    if args.threshold is not None and not np.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    if args.down_times is not None and any(factor <= 0 for factor in args.down_times):
        raise ValueError("down_times must contain positive integers")
    if args.h5_path and not os.path.isfile(args.h5_path):
        raise FileNotFoundError(f"HDF5 not found: {args.h5_path}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"\n{'=' * 60}\nRadio anomaly detection\n{'=' * 60}")
    print(f"Device: {device}")
    if args.h5_path:
        h5_path = os.path.abspath(args.h5_path)
        print(f"\n[1/4] Loading HDF5: {h5_path}")
    else:
        print("\n[1/4] Converting FITS inputs to HDF5")
        factors = args.down_times or list(DEFAULT_DOWN_TIMES)
        h5_path, _, _ = build_h5_from_fits_paths(
            [args.fits_dir],
            out_h5_path    = output_dir / "input_data.h5",
            down_time_list = factors,
            num_workers    = args.num_workers or 1,
        )

    if args.pred:
        dataset = H5PairDataset(h5_path, down_time_list=args.down_times)
        input_indices = np.asarray(
            [previous for previous, _ in dataset.pairs], dtype=np.int64
        )
        target_indices = np.asarray(
            [current for _, current in dataset.pairs], dtype=np.int64
        )
    else:
        with h5py.File(h5_path, "r") as h5:
            factors = np.asarray(h5["down_time"][:]).reshape(-1)
        if args.down_times is None:
            selected_indices = np.arange(factors.size, dtype=np.int64)
        else:
            selected_indices = np.flatnonzero(np.isin(factors, args.down_times))
        dataset        = H5RadioDataset(h5_path, indices=selected_indices)
        input_indices  = dataset.indices.copy()
        target_indices = dataset.indices.copy()
    print(f"Samples: {len(dataset)}")

    use_vae = args.model == "vae"
    print(f"\n[2/4] Loading model: {args.model_path}")
    if use_vae:
        model = VariationalRadioAutoencoder(
            pretrained   = False,
            latent_ch    = args.latent_dim,
            encoder_name = args.encoder,
        )
    else:
        model = RadioAutoencoder(
            pretrained   = False,
            encoder_name = args.encoder,
        )

    checkpoint = torch.load(
        args.model_path,
        map_location = device,
        weights_only = True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("model_config", {})
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
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    mode = f"PRED-{args.model.upper()}" if args.pred else args.model.upper()
    print(f"Model: {mode} + {args.encoder.upper()}")

    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = False,
        num_workers        = args.num_workers,
        pin_memory         = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
    print("\n[3/4] Computing anomaly scores")
    results = run_inference(
        model,
        loader,
        device,
        use_vae  = use_vae,
        use_pred = args.pred,
        beta     = args.kl_weight,
    )
    scores = results["scores"]

    with h5py.File(h5_path, "r") as h5:
        all_names_raw     = np.asarray(h5["names"][:]).reshape(-1)
        all_down_times    = np.asarray(h5["down_time"][:]).reshape(-1)
        all_slice_indices = np.asarray(h5["slice_idx"][:]).reshape(-1)
    all_names = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in all_names_raw
        ]
    )
    names         = all_names[target_indices]
    down_times    = all_down_times[target_indices].astype(np.int32)
    slice_indices = all_slice_indices[target_indices].astype(np.int32)

    print("\n[4/4] Saving results")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else compute_threshold(scores, args.threshold_method)
    )
    threshold_label = (
        "explicit" if args.threshold is not None else args.threshold_method
    )
    anomalies     = scores > threshold
    anomaly_count = int(anomalies.sum())
    print(f"Score mean: {scores.mean():.6f}; std: {scores.std():.6f}")
    print(f"Threshold ({threshold_label}): {threshold:.6f}")
    print(
        f"Anomalies: {anomaly_count}/{scores.size} "
        f"({100 * anomaly_count / scores.size:.1f}%)"
    )

    results_path = output_dir / "results.npz"
    np.savez(
        results_path,
        scores                = scores,
        reconstruction_scores = results["reconstruction_scores"],
        kl_scores             = results["kl_scores"],
        threshold             = threshold,
        anomalies             = anomalies,
        h5_indices            = target_indices,
        names                 = names,
        down_time             = down_times,
        slice_idx             = slice_indices,
    )
    print(f"Saved: {results_path}")
    plot_results(scores, threshold, output_dir)

    ranked_positions  = np.argsort(scores)[::-1]
    anomaly_list_path = output_dir / "anomalies.txt"
    with anomaly_list_path.open("w", encoding="utf-8") as output:
        output.write(f"# threshold\t{threshold:.8g}\n")
        output.write(f"# anomaly_count\t{anomaly_count}\n")
        output.write("# score\th5_index\tdown_time\tslice_idx\tname\n")
        for position in ranked_positions[anomalies[ranked_positions]]:
            output.write(
                f"{scores[position]:.8g}\t{target_indices[position]}\t"
                f"{down_times[position]}\t{slice_indices[position]}\t"
                f"{names[position]}\n"
            )
    print(f"Saved: {anomaly_list_path}")

    if args.plot_anomalies:
        anomaly_dir = output_dir / "anomaly_images"
        anomaly_dir.mkdir()
        plot_top_anomalies(
            model,
            h5_path,
            scores,
            input_indices,
            target_indices,
            names,
            down_times,
            slice_indices,
            args.top_n,
            anomaly_dir,
            device,
            use_vae  = use_vae,
            use_pred = args.pred,
        )

    print("\nHighest scores:")
    for rank, position in enumerate(ranked_positions[:10], start=1):
        marker = " anomaly" if anomalies[position] else ""
        print(
            f"  {rank}. {scores[position]:.6f} | "
            f"{Path(names[position]).name} | dt={down_times[position]} | "
            f"slice={slice_indices[position]}{marker}"
        )
    print(f"\nDone: {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--h5_path", help="Existing HDF5 dataset")
    inputs.add_argument(
        "--fits_dir",
        help="FITS file, directory, or glob pattern to convert before inference",
    )
    parser.add_argument("--down_times", type=int, nargs="+", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model", default="ae", choices=["ae", "vae"])
    parser.add_argument("--pred", action="store_true")
    parser.add_argument(
        "--encoder",
        default = "convnext",
        choices = ["convnext", "resnet18"],
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--threshold_method",
        default = "3sigma",
        choices = ["3sigma", "iqr", "percentile"],
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = None,
        help    = "Use a fixed calibrated threshold instead of estimating one",
    )
    parser.add_argument("--output_dir", default=str(BASE_DIR / "inference_results"))
    parser.add_argument("--plot_anomalies", action="store_true")
    parser.add_argument("--top_n", type=int, default=20)
    sys.exit(main(parser.parse_args()))

# fmt: on
