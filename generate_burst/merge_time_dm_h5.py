"""Merge generated time-DM CenterNet H5 shards into one H5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSONL（每行一个 JSON 对象）；文件不存在时返回空列表。"""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    """缩进 + key 排序写出单个 JSON（用于合并后的 config / inspect / 失败汇总）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """逐行写 JSONL（每张图一行合并后的 metadata）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_annotations_json(path: Path, image_count: int, ann: np.ndarray) -> None:
    """把合并后的 (N,5) annotations 还原成 per-image 标注 JSON；无框的图用占位行 [-1,-1,-1,-1]。"""
    by_image = {str(i): [[-1.0, -1.0, -1.0, -1.0]] for i in range(image_count)}
    for image_idx in range(image_count):
        boxes = ann[ann[:, 0] == image_idx, 1:5]            # 列 = [left, top, w, h]
        # 用 w/h>=1 过滤占位行；越界框 left/top 可为负，是合法标注，必须保留。
        boxes = boxes[(boxes[:, 2] >= 1) & (boxes[:, 3] >= 1)]
        if len(boxes):
            by_image[str(image_idx)] = [[float(v) for v in row] for row in boxes]
    path.write_text(json.dumps(by_image, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数：输出路径、多个输入 shard、gzip 等级、拷贝块大小、每图最大目标数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--copy-chunk", type=int, default=64)
    parser.add_argument("--max-objects-per-image", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """把多个 shard H5 顺序拼成一个：images/annotations 按偏移拼接（标注 image_idx 同步平移），
    合并 metadata/config，校验每图目标数不超限，并写出配套 annotations JSON 与 inspect 摘要。
    """
    args = parse_args()
    inputs = [path for path in args.inputs if path.exists()]
    if len(inputs) != len(args.inputs):
        missing = [str(path) for path in args.inputs if not path.exists()]
        raise SystemExit(f"Missing inputs: {missing}")
    if not inputs:
        raise SystemExit("No input H5 files")

    totals = []
    total_images = 0
    total_annotations = 0
    for path in inputs:
        with h5py.File(path, "r") as h5:
            image_count = int(h5["images"].shape[0])
            ann_count = int(h5["annotations"].shape[0])
            totals.append((path, image_count, ann_count))
            total_images += image_count
            total_annotations += ann_count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    compression = "gzip" if args.gzip_level > 0 else None
    opts = args.gzip_level if args.gzip_level > 0 else None
    str_dtype = h5py.string_dtype(encoding="utf-8")

    metadata_rows: list[dict] = []
    config_rows = []
    image_offset = 0
    ann_offset = 0
    with h5py.File(args.output, "w") as out:
        out.create_dataset(
            "images",
            shape=(total_images, 512, 512),
            dtype="float32",
            chunks=(1, 512, 512),
            compression=compression,
            compression_opts=opts,
        )
        out.create_dataset(
            "annotations",
            shape=(total_annotations, 5),
            dtype="float32",
            chunks=(min(max(total_annotations, 1), 4096), 5),
            compression=compression,
            compression_opts=opts,
        )
        out.create_dataset("original_filename", shape=(total_images,), dtype=str_dtype)
        out.create_dataset("original_path", shape=(total_images,), dtype=str_dtype)
        out.create_dataset("original_slice", shape=(total_images,), dtype="int32")
        out.attrs["generator"] = "merge_time_dm_h5.py"
        out.attrs["source_shards"] = len(inputs)

        for shard_index, (path, image_count, ann_count) in enumerate(totals):
            print(
                f"[merge] shard={shard_index} images={image_count} annotations={ann_count} path={path}",
                flush=True,
            )
            with h5py.File(path, "r") as h5:
                for start in range(0, image_count, args.copy_chunk):
                    stop = min(image_count, start + args.copy_chunk)
                    out["images"][image_offset + start:image_offset + stop] = h5["images"][start:stop]
                    out["original_filename"][image_offset + start:image_offset + stop] = h5["original_filename"][start:stop]
                    out["original_path"][image_offset + start:image_offset + stop] = h5["original_path"][start:stop]
                    out["original_slice"][image_offset + start:image_offset + stop] = h5["original_slice"][start:stop]

                ann = h5["annotations"][:]
                ann[:, 0] += image_offset
                out["annotations"][ann_offset:ann_offset + ann_count] = ann

            meta_path = path.with_suffix(".metadata.jsonl")
            for row in read_jsonl(meta_path):
                original_image_index = int(row.get("image_index", -1))
                if original_image_index >= 0:
                    row["source_image_index"] = original_image_index
                    row["image_index"] = image_offset + original_image_index
                row["shard_index"] = shard_index
                row["shard_h5"] = str(path)
                metadata_rows.append(row)

            cfg_path = path.with_suffix(".config.json")
            if cfg_path.exists():
                config_rows.append(json.loads(cfg_path.read_text(encoding="utf-8")))

            image_offset += image_count
            ann_offset += ann_count
            out.flush()

    with h5py.File(args.output, "r") as h5:
        ann = h5["annotations"][:]
        counts = np.bincount(ann[:, 0].astype(np.int64), minlength=total_images) if len(ann) else np.zeros(total_images, dtype=int)
        max_count = int(counts.max()) if len(counts) else 0
        bad = np.flatnonzero(counts > args.max_objects_per_image).astype(int).tolist()
        summary = {
            "output": str(args.output),
            "images": total_images,
            "annotations": total_annotations,
            "source_shards": len(inputs),
            "max_objects_per_image": max_count,
            "bad_images": bad[:100],
            "object_count_hist": {str(i): int((counts == i).sum()) for i in range(max_count + 1)},
            "metadata_rows": len(metadata_rows),
        }
    if bad:
        write_json(args.output.with_suffix(".merge_failed.json"), summary)
        raise SystemExit(f"Found {len(bad)} images with > {args.max_objects_per_image} objects")

    write_jsonl(args.output.with_suffix(".metadata.jsonl"), metadata_rows)
    merged_config = {
        "merge_summary": summary,
        "source_configs": config_rows,
    }
    if config_rows:
        first_rules = config_rows[0].get("annotation_rules")
        if first_rules is not None and all(row.get("annotation_rules") == first_rules for row in config_rows):
            merged_config["annotation_rules"] = first_rules
    write_json(args.output.with_suffix(".config.json"), merged_config)
    write_json(args.output.with_suffix(".inspect.json"), summary)
    write_annotations_json(args.output.with_name(f"{args.output.stem}_annotations.json"), total_images, ann)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
