"""Inspect generated time-DM CenterNet H5 files with a contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path | None) -> list[dict]:
    """逐行读取 JSONL 元信息；路径为空或文件不存在时返回空列表（元信息可选）。"""
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """1%/99.5% 分位截断后拉伸到 [0,1]，仅用于可视化显示。"""
    vmin, vmax = np.percentile(img, (1, 99.5))
    img = np.clip(img, vmin, vmax)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def choose_indices(total: int, count: int, seed: int) -> np.ndarray:
    """随机不重复抽 count 张图的索引（升序），用于拼接抽样预览图。"""
    count = min(total, count)
    if count <= 0:
        return np.empty((0,), dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def parse_args() -> argparse.Namespace:
    """解析命令行参数：H5 路径、可选元信息、抽样张数/随机种子、contact sheet 与 JSON 摘要输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """抽样若干张图，画出框 + 中心十字，拼成 contact sheet，并写一份 JSON 摘要做快速质检。"""
    args = parse_args()
    metadata_path = args.metadata or args.h5_path.with_suffix(".metadata.jsonl")
    meta_rows = read_jsonl(metadata_path)

    with h5py.File(args.h5_path, "r") as h5:
        total = int(h5["images"].shape[0])
        ann = h5["annotations"][:]
        indices = choose_indices(total, args.samples, args.seed)
        images = [h5["images"][int(i)][()] for i in indices]

    summary = {
        "h5_path": str(args.h5_path),
        "total_images": total,
        "sampled_indices": [int(i) for i in indices],
        "annotation_shape": list(ann.shape),
        "image_min": float(np.min([np.min(img) for img in images])) if images else None,
        "image_max": float(np.max([np.max(img) for img in images])) if images else None,
        "metadata_rows": len(meta_rows),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.contact_sheet is None or len(indices) == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    cols = min(6, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for panel, (idx, img) in enumerate(zip(indices, images)):
        ax = axes.ravel()[panel]
        ax.imshow(normalize_for_display(img), origin="lower", aspect="auto", cmap="viridis")
        boxes = ann[ann[:, 0].astype(np.int64) == int(idx), 1:5]
        for box_i, box in enumerate(boxes):
            x, y, w, h = [float(v) for v in box]
            cx = x + w / 2.0
            cy = y + h / 2.0
            color = "white" if box_i == 0 else "cyan"
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.0))
            ax.plot([cx], [cy], marker="+", color=color, markersize=9, markeredgewidth=1.5)
        title = f"idx={int(idx)}"
        if int(idx) < len(meta_rows):
            row = meta_rows[int(idx)]
            title += (
                f"\nS/N={row.get('snr', 0):.1f} DM={row.get('dm_pc_cm3', 0):.0f}"
                f" {row.get('split_name', '')}"
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 511)
        ax.set_ylim(0, 511)

    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=180)
    plt.close(fig)
    print(json.dumps({**summary, "contact_sheet": str(args.contact_sheet)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
