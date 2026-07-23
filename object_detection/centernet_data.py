"""H5 dataset + 数据增强 for CenterNet detector。

CenterNet 预测中心点（heatmap + 亚像素 offset），不回归宽高。本模块从 COCO 边框标注里取
中心点作为监督，并根据边框大小自适应决定 heatmap 高斯核半径。bbox 只提供中心和半径信息，
模型输出仍然只有中心点。

训练集做两层帧级过采样：
  1. 按一帧目标数（0/1/2-3/4+ → 1/3/5/8 倍）
  2. 含小/暗目标（最小 ``√(w·h) < SMALL_BOX_SIZE_THR``）的帧再额外 ×2

数据增强（仅训练集）：
  - 几何：仿射 + 翻转 + Resize。**bbox 不交给 albumentations**（2.x 会拒绝/裁剪越界框，
    裁剪后 ``x+w/2`` 会把监督中心带离真值）；而是把每个框的「中心 + 4 角点」当 keypoint
    跟踪（``remove_invisible=False``，允许出界）；增强后用中心 keypoint 重建真值中心、用
    4 角点重建 axis-aligned wh，再以真值中心为几何中心重组 COCO 框。因此框可以越界，
    但中心不会被 Albumentations 的 bbox 裁剪逻辑带偏；增强后据中心是否仍在画面内过滤目标。
  - 像素：噪声 / 模糊 / 亮度对比 / 局部遮挡
  - mosaic 拼图：2~5 张图按 rows×cols 拼成大图后 **mean-pool 下采样**回 imgsz²，每张源图整幅
    可见但被缩小，框等比缩小、中心随之正确缩放——以此扩充小/弱目标分布；触发概率见
    ``MOSAIC_PROB``。
"""

import os
import random

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# 1. 常量 / 全局配置
# ---------------------------------------------------------------------------

_EMPTY_BOXES = np.empty((0, 4), dtype=np.float64)

# 训练集过采样：√(w·h) 小于该值的框视为「小/暗目标」，对应帧额外加倍采样
SMALL_BOX_SIZE_THR = 32.0
SMALL_BOX_EXTRA_REPEAT = 2

# mosaic 拼图触发概率（仅训练集；验证集恒为 0）
MOSAIC_PROB = 0.5

# (名称, 一帧目标数谓词, 基础复制倍数)
_BOX_COUNT_BUCKETS = [
    ("0",   lambda c: c == 0,             1),
    ("1",   lambda c: c == 1,             3),
    ("2-3", lambda c: 2 <= c <= 3,        5),
    ("4+",  lambda c: c >= 4,             8),
]


# ---------------------------------------------------------------------------
# 2. 通用工具：图像归一化、albumentations 兼容、框过滤
# ---------------------------------------------------------------------------

def normalize_image(img):
    """按 1% / 99.5% 分位截断后线性拉伸到 [0, 1]（float32）。"""
    vmin, vmax = np.percentile(img, (1, 99.5))
    img = np.clip(img, vmin, vmax)
    return ((img - img.min()) / (img.max() - img.min() + np.float32(1e-8))).astype(np.float32)


def _keypoint_params(coord_format, **kwargs):
    """构造 KeypointParams，兼容新/老 albumentations 的参数名差异（``format`` vs ``coord_format``）。"""
    try:
        return A.KeypointParams(format=coord_format, **kwargs)
    except TypeError:
        return A.KeypointParams(coord_format=coord_format, **kwargs)


def _filter_valid_boxes(bboxes):
    """保留有限数值且宽高 >= 1 px 的 COCO 框；其它输入统一返回空框数组 ``[0, 4]``。"""
    if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and len(bboxes) > 0:
        bboxes = bboxes.astype(np.float64, copy=False)
        keep = np.isfinite(bboxes).all(axis=1) & (bboxes[:, 2] >= 1) & (bboxes[:, 3] >= 1)
        return bboxes[keep]
    return _EMPTY_BOXES.copy()


# ---------------------------------------------------------------------------
# 3. 数据切分 + 训练集过采样
# ---------------------------------------------------------------------------

def _box_count_repeat(box_count):
    for _, pred, repeat in _BOX_COUNT_BUCKETS:
        if pred(box_count):
            return repeat
    return 1


def _box_count_summary(df):
    return ", ".join(
        f"{name}={int(df['box_count'].map(pred).sum())}" for name, pred, _ in _BOX_COUNT_BUCKETS
    )


def _oversample_train_df(train_df, seed):
    """按目标数 + 是否含小目标对训练集做帧级过采样，并打印 before / after 分布。"""
    base = train_df["box_count"].map(_box_count_repeat).to_numpy()
    has_small = (train_df["min_box_size"] < SMALL_BOX_SIZE_THR).to_numpy()
    repeats = base * np.where(has_small, SMALL_BOX_EXTRA_REPEAT, 1)

    before = _box_count_summary(train_df)
    small_before = int(has_small.sum())

    expanded = train_df.loc[train_df.index.repeat(repeats)]
    expanded = expanded.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    after = _box_count_summary(expanded)
    small_after = int((expanded["min_box_size"] < SMALL_BOX_SIZE_THR).sum())

    print(f"[Data] Repeat: box-count 0:1, 1:3, 2-3:5, 4+:8  |"
          f"  small-box (<{SMALL_BOX_SIZE_THR:g}px) extra x{SMALL_BOX_EXTRA_REPEAT}")
    print(f"[Data] Train box-count before->after: [{before}] -> [{after}]")
    print(f"[Data] Train small-box frames before->after: {small_before} -> {small_after}")
    return expanded


def get_train_val(data_folder, train_ratio=0.8, seed=42):
    """扫描 ``data_folder`` 下所有 ``.h5``，按帧建表并切分 train / val。

    切分先于过采样完成，保证 train/val 没有泄漏。每行对应一帧；``min_box_size`` 用于过采样判定。
    """
    h5_files = sorted(
        os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".h5")
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in: {data_folder}")

    records = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            total_images = int(f["images"].shape[0])
            all_ann = f["annotations"][:]
        for img_idx in range(total_images):
            img_ann = all_ann[all_ann[:, 0] == img_idx, 1:]      # 列 = [left, top, w, h]
            # 只按数值有限和 w/h 过滤占位行。不能用 left/top>=0：越界框允许从图外开始。
            bboxes = _filter_valid_boxes(img_ann)
            min_box_size = (
                float(np.sqrt(bboxes[:, 2] * bboxes[:, 3]).min()) if len(bboxes) else float("inf")
            )
            records.append({
                "h5_path": h5_path,
                "img_idx": int(img_idx),
                "bboxes": bboxes,
                "box_count": int(len(bboxes)),
                "has_annotation": len(bboxes) > 0,
                "min_box_size": min_box_size,
            })

    df = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    val_df = df[split_idx:].reset_index(drop=True)
    train_df = _oversample_train_df(df[:split_idx].reset_index(drop=True), seed)

    train_pos = int(train_df["has_annotation"].sum())
    val_pos = int(val_df["has_annotation"].sum())
    print(f"[Data] Train: {len(train_df)} ({train_pos} pos / {len(train_df) - train_pos} neg)"
          f" | Val: {len(val_df)} ({val_pos} pos / {len(val_df) - val_pos} neg)")
    return train_df, val_df


# ---------------------------------------------------------------------------
# 4. CenterNet 监督目标生成：高斯核 heatmap + 亚像素 offset
# ---------------------------------------------------------------------------

def gaussian2d(shape, sigma=1):
    """以原点为中心、尺寸 ``shape`` 的 2D 高斯核。"""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    """CenterNet 标准的自适应半径：保证半径内的预测框与 GT 仍能达到 ``min_overlap`` IoU。

    ``det_size`` 为 (h, w)，单位是 heatmap 像素（即除过 down_ratio 之后的尺寸）。
    """
    h, w = det_size
    a1, b1, c1 = 1.0, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)
    a2, b2, c2 = 4.0, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def draw_umich_gaussian(heatmap, center, radius):
    """把以 ``center`` 为中心、半径 ``radius`` 的高斯核 element-wise max 到 ``heatmap`` 上。"""
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


# ---------------------------------------------------------------------------
# 5. Dataset
# ---------------------------------------------------------------------------

class H5CenterNetDataset(torch.utils.data.Dataset):
    """逐帧产出 ``{img, hm, reg_mask, ind, offset, gt_centers, meta}`` 的 CenterNet 训练样本。

    - ``img``        : ``[1, imgsz, imgsz]`` float32，灰度
    - ``hm``         : ``[1, out, out]``，中心 heatmap（自适应高斯）
    - ``reg_mask``   : ``[max_objs]``，有效目标位置 mask
    - ``ind``        : ``[max_objs]``，目标在 heatmap 展平索引（``y * out + x``）
    - ``offset``     : ``[max_objs, 2]``，中心点亚像素 offset（(cx - int(cx), cy - int(cy))）
    - ``gt_centers`` : ``[N, 2]`` 原图坐标系下的真值中心，仅用于评估
    - ``meta``       : h5 路径 + 帧 index，便于调试 / 推理

    单帧处理流水线（见 :meth:`__getitem__`）：
        读盘 / mosaic 拼图 → 过滤无效框 → 归一化 → 增强（仅训练）→ 再归一化 → 生成监督目标
    """

    _KP_PER_BOX = 5   # 每个框跟踪的 keypoint 数：中心 + 4 角点

    def __init__(self, dataframe, imgsz=512, down_ratio=4, max_objs=128,
                 center_radius=3, val=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.imgsz = int(imgsz)
        self.down_ratio = int(down_ratio)
        self.output_size = self.imgsz // self.down_ratio
        self.max_objs = int(max_objs)
        self.min_radius = max(int(center_radius), 1)   # 小目标也至少给 1 像素半径，保证监督信号
        self.val = bool(val)
        self.mosaic_prob = 0.0 if self.val else MOSAIC_PROB
        self.h5_files = {}                              # worker-local h5 文件句柄缓存
        self._build_augmentations()

    # ---- 增强 pipeline 构造（__init__ 时一次性建好）------------------------

    def _build_augmentations(self):
        """构造增强 pipeline：几何（keypoint 跟踪中心+角点）+ 像素，两条 ``A.Compose``。"""
        # 几何增强：bow-tie 目标可见尺寸差异大，尺度抖动是主要几何增强
        geometric = [
            A.Affine(
                scale={"x": (0.3, 2.0), "y": (0.3, 2.0)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                keep_ratio=False,
                p=0.7,
            ),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Resize(height=self.imgsz, width=self.imgsz, p=1.0),
        ]
        # 像素增强：噪声、模糊、亮度对比、小块遮挡
        pixel = [
            A.Downscale(scale_range=(0.4, 0.9), p=0.20),
            A.OneOf([
                A.GaussNoise(std_range=(0, 0.2), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),
            A.CoarseDropout(num_holes_range=(1, 6),
                            hole_height_range=(8, 32), hole_width_range=(8, 32),
                            fill=0, p=0.25),
        ]

        # 几何增强只跟踪 keypoint：每个框拆成「中心 + 4 角点」，remove_invisible=False
        # 允许出界，增强后再从 keypoint 重建中心和 wh（见 _augment）。不挂 bbox_params，
        # 避免 albumentations 2.x 对越界框报错 / 裁剪导致中心漂移。
        self._geometric_aug = A.Compose(
            geometric,
            keypoint_params=_keypoint_params(
                "xy", remove_invisible=False, label_fields=["keypoint_ids"], label_mapping={},
            ),
        )
        # 像素增强不影响坐标，单独成一条 Compose（纯像素变换，不挂 keypoint_params）
        self._pixel_aug = A.Compose(pixel)

    # ---- 主入口 ------------------------------------------------------------

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        if not self.val and random.random() < self.mosaic_prob:
            img, bboxes = self._load_mosaic(idx)
        else:
            img, bboxes = self._load_single(idx)

        bboxes = _filter_valid_boxes(bboxes)
        img = normalize_image(img)
        if not self.val:
            img, bboxes = self._augment(img, bboxes)
        img = normalize_image(img)                          # 增强后再归一一次，修正值域偏移

        targets = self._build_targets(bboxes)
        return {
            "img": torch.from_numpy(img).float().unsqueeze(0).contiguous(),     # [1, H, W]
            "hm": torch.from_numpy(targets["hm"]).float(),
            "reg_mask": torch.from_numpy(targets["reg_mask"]).float(),
            "ind": torch.from_numpy(targets["ind"]).long(),
            "offset": torch.from_numpy(targets["offset"]).float(),
            "gt_centers": torch.from_numpy(self._coco_to_centers(bboxes)).float(),
            "meta": {"h5_path": row["h5_path"], "img_idx": int(row["img_idx"])},
        }

    # ---- 读盘 + mosaic 拼图 ------------------------------------------------

    def _load_single(self, idx):
        row = self.dataframe.iloc[idx]
        h5_path = row["h5_path"]
        if h5_path not in self.h5_files:
            self.h5_files[h5_path] = h5py.File(h5_path, "r")        # 每个 worker 独立持有句柄
        img = self.h5_files[h5_path]["images"][row["img_idx"]].copy()
        bboxes = row["bboxes"].copy() if isinstance(row["bboxes"], np.ndarray) else row["bboxes"]
        return img, bboxes

    def _load_mosaic(self, idx):
        """随机决定拼图模式（2×2 / 1×n / n×1），输出仍是 imgsz² 单图。"""
        comb_num = np.random.randint(1, 6)
        if comb_num == 1:
            return self._load_single(idx)

        # 小型 smoke 数据集可能少于 comb_num 帧；允许重复采样，避免调试时直接报错。
        other = np.random.choice(
            len(self.dataframe), comb_num - 1, replace=len(self.dataframe) < comb_num,
        )
        indices = np.append([idx], other)
        imgs, boxes_list = [], []
        for i in indices:
            img, bboxes = self._load_single(i)
            imgs.append(img)
            boxes_list.append(bboxes)

        # 4 张图有一半概率走 2×2 网格，否则按水平 / 垂直条带；条带在另一方向上保持单层
        if comb_num == 4 and np.random.rand() > 0.5:
            rows, cols = 2, 2
        elif np.random.rand() > 0.5:
            rows, cols = 1, comb_num
        else:
            rows, cols = comb_num, 1
        return self._mosaic(imgs, boxes_list, rows, cols, self.imgsz)

    @staticmethod
    def _mosaic(imgs, boxes_list, rows, cols, imgsz):
        """把 ``rows*cols`` 张 ``imgsz²`` 图拼成 (imgsz·rows, imgsz·cols)，再 mean-pool 压回 imgsz²。

        与"裁 tile"不同，这里每张源图**整幅**放进格子，再按格子块平均下采样回 imgsz²：
        每个源图被缩小（横向 cols 倍、纵向 rows 倍），其中所有目标都可见但更小、更弱——
        正是用来扩充小/弱目标分布。所有框等比缩小（x,w /= cols；y,h /= rows），中心随之
        正确缩放（``x+w/2`` 仍等于缩放后的真值中心），越界框也只是按比例缩小，照样兼容
        后续 keypoint 重建。

        要求每张源图恰为 imgsz²（本数据集即 512²）。框不在此过滤，留给 __getitem__ 统一处理。
        """
        imgsz = int(imgsz)
        canvas = np.zeros((imgsz * rows, imgsz * cols), dtype=np.float32)
        bbox_data = []
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            canvas[imgsz * r: imgsz * (r + 1), imgsz * c: imgsz * (c + 1)] = imgs[i]
            if isinstance(boxes_list[i], np.ndarray) and len(boxes_list[i]) > 0:
                b = boxes_list[i].astype(np.float64).copy()
                b[:, 0] += imgsz * c                                # x 偏移到该格
                b[:, 1] += imgsz * r                                # y 偏移到该格
                bbox_data.append(b)

        # reshape→mean 实现 (rows, cols) 块平均下采样（轴排布见 reshape((imgsz, rows, imgsz, cols))）
        img = canvas.reshape(imgsz, rows, imgsz, cols).mean(axis=(1, 3))
        if bbox_data:
            boxes = np.vstack(bbox_data)
            boxes[:, [0, 2]] /= cols                                # x, w 同步缩小
            boxes[:, [1, 3]] /= rows                                # y, h 同步缩小
        else:
            boxes = _EMPTY_BOXES.copy()
        return img, boxes

    # ---- 数据增强 ----------------------------------------------------------

    def _augment(self, img, bboxes):
        """几何增强用 keypoint 跟踪每个框的中心 + 4 角点，增强后重建框；再跑像素增强。

        关键：bbox 不进 albumentations，避免 2.x 拒绝越界框 / 裁剪导致中心漂移。框被拆成
        keypoint（中心 + 4 角点，remove_invisible=False 允许出界）同步几何变换；增强后用
        中心 keypoint 取真值中心、用 4 角点取 axis-aligned wh，再以真值中心为几何中心重组框
        （可越界）。无框样本传空 keypoint，走同一条 pipeline。
        """
        keypoints, kp_ids = self._boxes_to_keypoints(bboxes)
        augmented = self._geometric_aug(image=img, keypoints=keypoints, keypoint_ids=kp_ids)
        img_out = self._pixel_aug(image=augmented["image"])["image"]
        boxes_out = self._keypoints_to_boxes(
            augmented.get("keypoints", []), augmented.get("keypoint_ids", []),
        )
        return img_out, boxes_out

    def _boxes_to_keypoints(self, bboxes):
        """每个 COCO 框 → 5 个 keypoint：中心 + 4 角点；keypoint_id = box_index*5 + role。"""
        if not (isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and len(bboxes) > 0):
            return [], []
        keypoints, kp_ids = [], []
        for bi, (x, y, w, h) in enumerate(bboxes):
            cx, cy = x + w / 2.0, y + h / 2.0
            pts = [
                (cx, cy),            # role 0: 中心（真值监督）
                (x, y),              # role 1: 左上
                (x + w, y),          # role 2: 右上
                (x + w, y + h),      # role 3: 右下
                (x, y + h),          # role 4: 左下
            ]
            for role, (px, py) in enumerate(pts):
                keypoints.append((float(px), float(py)))
                kp_ids.append(bi * self._KP_PER_BOX + role)
        return keypoints, kp_ids

    def _keypoints_to_boxes(self, keypoints, kp_ids):
        """增强后的 keypoint → COCO 框：中心取真值 keypoint、wh 取 4 角点 axis-aligned 跨度。

        中心移出画面的目标视为不可见，丢弃。框以真值中心为几何中心重组，允许越界；
        ``_build_targets`` 后续只从它取中心和 heatmap 半径，不做 bbox 回归。
        """
        groups = {}
        for kp, kid in zip(keypoints, kp_ids):
            bi, role = divmod(int(kid), self._KP_PER_BOX)
            groups.setdefault(bi, {})[role] = (float(kp[0]), float(kp[1]))

        boxes = []
        for bi in sorted(groups):
            roles = groups[bi]
            if 0 not in roles:
                continue
            cx, cy = roles[0]
            if not (0.0 <= cx < self.imgsz and 0.0 <= cy < self.imgsz):
                continue                              # 中心出界 → 目标不可见
            corners = [roles[r] for r in (1, 2, 3, 4) if r in roles]
            if len(corners) == 4:
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                w = max(max(xs) - min(xs), 1.0)
                h = max(max(ys) - min(ys), 1.0)
            else:
                w = h = 1.0
            boxes.append([cx - w / 2.0, cy - h / 2.0, w, h])

        if not boxes:
            return _EMPTY_BOXES.copy()
        return np.array(boxes, dtype=np.float64)

    # ---- 目标生成 ----------------------------------------------------------

    def _build_targets(self, bboxes):
        """COCO bbox → CenterNet 监督：中心 heatmap（自适应高斯）+ 亚像素 offset。

        高斯半径用 :func:`gaussian_radius` 按目标在 heatmap 上的尺寸算，并以
        ``min_radius`` 兜底——大目标更宽，小目标至少给 ``min_radius`` 像素以保证有梯度。
        """
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs,), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        offset = np.zeros((self.max_objs, 2), dtype=np.float32)

        if not isinstance(bboxes, np.ndarray) or len(bboxes) == 0:
            return {"hm": hm, "reg_mask": reg_mask, "ind": ind, "offset": offset}

        n = 0
        for x, y, w, h in bboxes:
            cx = (x + w / 2.0) / self.down_ratio
            cy = (y + h / 2.0) / self.down_ratio
            if not (0 <= cx < self.output_size and 0 <= cy < self.output_size):
                continue

            box_h = max(h / self.down_ratio, 1.0)
            box_w = max(w / self.down_ratio, 1.0)
            radius = max(int(round(gaussian_radius((box_h, box_w)))), self.min_radius)

            ct_int = np.array([cx, cy], dtype=np.float32).astype(np.int32)
            draw_umich_gaussian(hm[0], ct_int, radius)
            ind[n] = int(ct_int[1] * self.output_size + ct_int[0])
            offset[n] = np.array([cx - ct_int[0], cy - ct_int[1]], dtype=np.float32)
            reg_mask[n] = 1.0
            n += 1
            if n >= self.max_objs:
                break

        return {"hm": hm, "reg_mask": reg_mask, "ind": ind, "offset": offset}

    def _coco_to_centers(self, bboxes):
        """COCO bbox → 原图坐标系下的真值中心 [N, 2]（仅用于评估）。"""
        if not isinstance(bboxes, np.ndarray) or len(bboxes) == 0:
            return np.empty((0, 2), dtype=np.float32)
        x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        centers = np.stack([x + w / 2.0, y + h / 2.0], axis=1)
        keep = (
            (centers[:, 0] >= 0) & (centers[:, 0] < self.imgsz)
            & (centers[:, 1] >= 0) & (centers[:, 1] < self.imgsz)
        )
        return centers[keep].astype(np.float32)

    # ---- 资源清理 ----------------------------------------------------------

    def close(self):
        for h5f in self.h5_files.values():
            try:
                h5f.close()
            except Exception:
                pass
        self.h5_files.clear()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# 6. Collate fn：把样本 dict 列表拼成 batch dict
# ---------------------------------------------------------------------------

def centernet_collate_fn(batch):
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "hm": torch.stack([b["hm"] for b in batch]),
        "reg_mask": torch.stack([b["reg_mask"] for b in batch]),
        "ind": torch.stack([b["ind"] for b in batch]),
        "offset": torch.stack([b["offset"] for b in batch]),
        "gt_centers": [b["gt_centers"] for b in batch],   # 变长，不能 stack
        "meta": [b["meta"] for b in batch],
    }
