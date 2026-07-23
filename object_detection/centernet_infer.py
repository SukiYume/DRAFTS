"""在验证集上可视化 CenterNet 的预测（GT 绿、预测红）。

可视化没有框（CenterNet 只预测中心点），所以 GT 和预测都用「空心十字」标出中心位置。
"""

import argparse
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from centernet_data import get_train_val, normalize_image
from centernet_eval import decode_centernet_outputs
from centernet_model import build_centernet_model


# ---------------------------------------------------------------------------
# 1. 设备 + 模型加载
# ---------------------------------------------------------------------------

def _pick_device(args):
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.device:
        return torch.device(f"cuda:{args.device}")
    return torch.device("cuda")


def _load_model(args, device):
    """构建模型并加载权重；同时兼容 ``state_dict`` 和完整 checkpoint dict。"""
    log_name = args.backbone.replace("convnext_", "conv_")
    weights = args.weights or os.path.join("./logs", f"logs_centernet_{log_name}", "best_model_ema.pth")
    model = build_centernet_model(
        backbone=args.backbone, pretrained=False,
        down_ratio=args.down_ratio, head_ch=args.head_ch,
    )
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# 2. 单图推理
# ---------------------------------------------------------------------------

def _predict_single(model, img, device, args):
    """img 是已归一化的灰度图 [H, W]；返回 (scores_np, centers_np) 或 (None, None)。"""
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        outputs = model(img_t)
    return decode_centernet_outputs(
        outputs, conf_thr=args.conf, topk=args.topk, down_ratio=args.down_ratio,
    )[0]


# ---------------------------------------------------------------------------
# 3. 可视化
# ---------------------------------------------------------------------------

def _draw_hollow_cross(vis, center, color, arm=9, gap=3, thickness=1):
    """中心留空的十字 + 小圆圈；适合给点状目标做可视化。"""
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.line(vis, (x - arm, y), (x - gap, y), color, thickness)
    cv2.line(vis, (x + gap, y), (x + arm, y), color, thickness)
    cv2.line(vis, (x, y - arm), (x, y - gap), color, thickness)
    cv2.line(vis, (x, y + gap), (x, y + arm), color, thickness)
    cv2.circle(vis, (x, y), gap, color, thickness)


def _draw_gt(vis, bboxes):
    """从 COCO bbox 取中心，画绿色十字。无标注时直接返回。"""
    if not isinstance(bboxes, np.ndarray) or bboxes.ndim != 2 or len(bboxes) == 0:
        return
    for x, y, w, h in bboxes:
        _draw_hollow_cross(vis, (x + w / 2.0, y + h / 2.0), (0, 255, 0))


def _draw_pred(vis, scores, centers, sample_idx):
    """预测红色十字；并打印每个预测的 score / 坐标。"""
    if centers is None or scores is None:
        print(f"sample={sample_idx} pred=0")
        return
    for ci, center in enumerate(centers):
        _draw_hollow_cross(vis, center, (0, 0, 255))
        print(f"sample={sample_idx} pred={ci} score={scores[ci]:.4f} center=({center[0]:.1f},{center[1]:.1f})")


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

def main(args):
    device = _pick_device(args)
    model = _load_model(args, device)

    # 用同样的 seed/train_ratio 复现训练时的 val 划分
    _, val_df = get_train_val(args.data_path, train_ratio=args.train_ratio, seed=args.seed)
    preview_indices = range(args.start, min(args.end, len(val_df)))

    h5_cache = {}                                       # 进程内复用文件句柄，避免反复打开
    for sample_idx in preview_indices:
        row = val_df.iloc[sample_idx]
        h5_path = row["h5_path"]
        if h5_path not in h5_cache:
            h5_cache[h5_path] = h5py.File(h5_path, "r")

        img = normalize_image(h5_cache[h5_path]["images"][row["img_idx"]].copy())
        scores, centers = _predict_single(model, img, device, args)

        vis = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        _draw_gt(vis, row["bboxes"])
        _draw_pred(vis, scores, centers, sample_idx)

        plt.figure(figsize=(5, 5))
        plt.title(f"CenterNet Validation Sample {sample_idx}")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))  # OpenCV BGR → matplotlib RGB
        plt.axis("off")
        plt.show()

    for f in h5_cache.values():
        f.close()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "convnext_tiny", "convnext_small"])   # 须与训练时一致
    parser.add_argument("--head-ch", type=int, default=128)
    parser.add_argument("--down-ratio", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=30)
    main(parser.parse_args())
