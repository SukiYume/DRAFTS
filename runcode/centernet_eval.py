"""CenterNet 输出解码 + 基于中心距离的 P / R / F1 / 中心偏差评估。

CenterNet 预测的是「中心点」，没有宽高，所以匹配本来就用欧氏距离（不是 IoU）：
预测中心到最近未占用 GT 中心的距离 ≤ ``dist_thr`` 即记为 TP。

最终训练时用来挑 best checkpoint 的择优分数（越大越好）：

    center_quality = max(0, 1 - p90_center_dist / dist_thr)
    score = 0.50 × recall + 0.20 × precision + 0.30 × center_quality

其中中心偏差在 score 选中的置信度工作点上统计。检测阶段偏召回，假阳性后续由分类器过滤。
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# score 权重：检测阶段偏召回；中心 p90 偏差约束后续原始数据提取是否还能对准。
RECALL_WEIGHT = 0.50
PRECISION_WEIGHT = 0.20
CENTER_QUALITY_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# 1. 解码：heatmap → 候选中心 (scores, centers)
# ---------------------------------------------------------------------------

def _nms_heatmap(hm, kernel=3):
    """3×3 max-pool 做峰值检测：只保留等于局部最大值的位置（heatmap 版 NMS）。"""
    pad = kernel // 2
    hmax = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    return hm * (hmax == hm).float()


def _topk(scores, k):
    """单类 heatmap，每张图取分数最高的 k 个像素 → (top_scores, flat_inds, xs, ys)。"""
    b, _, h, w = scores.shape
    k = min(k, h * w)
    top_scores, top_inds = torch.topk(scores.view(b, -1), k)
    return top_scores, top_inds, (top_inds % w).float(), (top_inds // w).float()


def _gather_offset(offset, inds):
    """[B, 2, H, W] 按展平索引 inds ([B, K]) 取出对应位置 → [B, K, 2]。"""
    b, c, h, w = offset.shape
    flat = offset.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    return flat.gather(1, inds.unsqueeze(2).expand(b, inds.size(1), c))


@torch.no_grad()
def decode_centernet_outputs(outputs, conf_thr=0.1, topk=100, down_ratio=4):
    """模型原始输出 → 每图 ``(scores_np, centers_np)``；centers 已乘 down_ratio 回到原图坐标系。

    Returns:
        list[tuple]: 长度 = batch；某图无检测时为 ``(None, None)``。
    """
    hm = _nms_heatmap(torch.sigmoid(outputs["hm"]))
    scores, inds, xs, ys = _topk(hm, k=topk)
    offsets = _gather_offset(outputs["offset"], inds)
    xs = (xs + offsets[..., 0]) * down_ratio   # 整数格点 + 亚像素 offset，乘以 stride 回原图
    ys = (ys + offsets[..., 1]) * down_ratio

    results = []
    for b in range(scores.size(0)):
        keep = scores[b] > conf_thr
        if not keep.any():
            results.append((None, None))
            continue
        centers = torch.stack([xs[b][keep], ys[b][keep]], dim=1)
        results.append((scores[b][keep].cpu().numpy(), centers.cpu().numpy()))
    return results


# ---------------------------------------------------------------------------
# 2. 单图贪心匹配：[n_pred, n_gt] 距离矩阵 → 每个预测的 TP 标记 + 匹配距离
# ---------------------------------------------------------------------------

def _greedy_match(dist_matrix, dist_thr):
    """按预测顺序贪心匹配 GT；挑距离最小的 GT，需 ≤ dist_thr 且未被占用 → TP。

    Args:
        dist_matrix: numpy ``[n_pred, n_gt]``，预测 i 到 GT j 的欧氏距离。
        dist_thr:    判定为 TP 的距离阈值（像素）。

    Returns:
        tp:   ``[n_pred]`` 0/1
        dist: ``[n_pred]`` 匹配距离；FP 处为 NaN
    """
    n_pred, n_gt = dist_matrix.shape
    tp = np.zeros(n_pred, dtype=np.float32)
    matched_dist = np.full(n_pred, np.nan, dtype=np.float32)
    if n_pred == 0 or n_gt == 0:
        return tp, matched_dist

    taken = set()
    for pi in range(n_pred):
        gi = int(np.argmin(dist_matrix[pi]))
        best = float(dist_matrix[pi, gi])
        if best <= dist_thr and gi not in taken:
            tp[pi] = 1.0
            matched_dist[pi] = best
            taken.add(gi)
    return tp, matched_dist


# ---------------------------------------------------------------------------
# 3. PR 曲线 + 距离统计 + 主聚合
# ---------------------------------------------------------------------------

def _pr_curve(all_tp, all_conf, total_gt, all_dist=None):
    """按 conf 降序展开整集 PR 曲线。空输入返回长度 0 的数组，调用方自行处理。

    Returns:
        precision, recall, f1, conf, dist_sorted (后者可能为 None)
    """
    if not all_tp or total_gt == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, None

    tp = np.concatenate(all_tp)
    conf = np.concatenate(all_conf)
    dist = np.concatenate(all_dist) if all_dist is not None else None
    order = np.argsort(-conf)
    tp, conf = tp[order], conf[order]
    if dist is not None:
        dist = dist[order]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, conf, dist


def _distance_stats(dist_values, dist_thr):
    """返回中心偏差统计；无 TP 时退化为最坏情况。"""
    if dist_values.size == 0:
        worst = float(dist_thr)
        return {
            "mean_center_dist": worst,
            "p50_center_dist": worst,
            "p90_center_dist": worst,
            "p95_center_dist": worst,
            "center_quality": 0.0,
        }

    p50, p90, p95 = (float(np.percentile(dist_values, q)) for q in (50, 90, 95))
    return {
        "mean_center_dist": float(np.mean(dist_values)),
        "p50_center_dist": p50,
        "p90_center_dist": p90,
        "p95_center_dist": p95,
        "center_quality": max(0.0, 1.0 - p90 / float(dist_thr)),
    }


def _compute_pr_metrics(all_tp, all_conf, all_dist, total_gt, dist_thr):
    """在召回优先的工作点上算 P/R/F1，再叠加中心偏差统计与最终 score。"""
    empty = {
        "f1": 0.0, "f1_conf": 0.0,
        "precision": 0.0, "recall": 0.0,
        "mean_center_dist": float(dist_thr),
        "p50_center_dist": float(dist_thr),
        "p90_center_dist": float(dist_thr),
        "p95_center_dist": float(dist_thr),
        "center_quality": 0.0,
        "score": 0.0,
    }
    precision, recall, f1, conf, dist = _pr_curve(all_tp, all_conf, total_gt, all_dist)
    if precision.size == 0:
        return empty

    # 检测阶段漏检不可恢复，误报还能交给分类器过滤，所以工作点选择偏召回
    detection_score = RECALL_WEIGHT * recall + PRECISION_WEIGHT * precision
    best = int(np.argmax(detection_score))

    # 中心偏差只在 score 选中的同一置信度工作点上统计，避免阈值口径不一致
    selected_dist = dist[:best + 1]
    dist_stats = _distance_stats(selected_dist[~np.isnan(selected_dist)], dist_thr)

    precision_v = float(precision[best])
    recall_v = float(recall[best])
    return {
        "f1": float(f1[best]),
        "f1_conf": float(conf[best]),
        "precision": precision_v,
        "recall": recall_v,
        **dist_stats,
        "score": (
            RECALL_WEIGHT * recall_v
            + PRECISION_WEIGHT * precision_v
            + CENTER_QUALITY_WEIGHT * dist_stats["center_quality"]
        ),
    }


# ---------------------------------------------------------------------------
# 4. 主入口
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(model, loader, device, conf_thr=0.01, topk=100, down_ratio=4, dist_thr=8.0):
    """在验证集上算 P / R / F1 / 中心偏差分位数 / score。

    Args:
        model: 推理用模型（一般传 EMA），已在 ``device`` 上。
        loader: 非分布式的验证 DataLoader（rank 0 独占）。
        conf_thr: 候选过滤阈值；用低阈值以获得完整的 PR 曲线。
        topk: 每图保留的最大候选数。
        down_ratio: heatmap 步长，用于把 (x,y) 还原到原图坐标。
        dist_thr: 中心距离匹配阈值（原图像素）；同时用作 score 公式的归一化分母。

    Returns:
        dict: ``{"score", "f1", "f1_conf", "precision", "recall", "center_quality",
        "mean_center_dist", "p50_center_dist", "p90_center_dist", "p95_center_dist", "dist_thr"}``
        score 越大越好，训练里直接用 score 挑 best checkpoint。
    """
    model.eval()
    all_tp, all_conf, all_dist, total_gt = [], [], [], 0

    for batch in tqdm(loader, dynamic_ncols=True, ascii=True, desc="center eval"):
        imgs = batch["img"].to(device, non_blocking=True)
        outputs = model(imgs)
        decoded = decode_centernet_outputs(outputs, conf_thr=conf_thr, topk=topk, down_ratio=down_ratio)

        for (scores, centers), gt_centers_t in zip(decoded, batch["gt_centers"]):
            gt_centers = gt_centers_t.cpu().numpy()
            total_gt += len(gt_centers)
            if scores is None:
                continue

            # 先把 [n_pred, n_gt] 距离矩阵算出来；再交给 _greedy_match 做匹配
            if len(gt_centers) > 0:
                diff = centers[:, None, :] - gt_centers[None, :, :]
                dist_mat = np.sqrt((diff ** 2).sum(axis=2))
                tp, dist = _greedy_match(dist_mat, dist_thr)
            else:
                tp = np.zeros(len(centers), dtype=np.float32)
                dist = np.full(len(centers), np.nan, dtype=np.float32)

            all_tp.append(tp)
            all_conf.append(scores)
            all_dist.append(dist)

    metrics = _compute_pr_metrics(all_tp, all_conf, all_dist, total_gt, dist_thr=dist_thr)
    metrics["dist_thr"] = float(dist_thr)
    return metrics
