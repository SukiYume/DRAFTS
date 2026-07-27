"""Center-only (CenterNet) detector：torchvision backbone + 多尺度融合到 stride-4 + 中心头。

提供 backbone（``--backbone`` 传简称即可）：
  * ``resnet18``      : :func:`torchvision.models.resnet18`，3×3 卷积，GPU 吞吐高、TensorRT 友好。
  * ``convnext_tiny`` : :func:`torchvision.models.convnext_tiny`，与下游 ConvNeXt 分类器同系。
  * ``convnext_small``: :func:`torchvision.models.convnext_small`，更深的 ConvNeXt 版本。

ImageNet 预训练 stem conv 的 3 通道权重会按通道取均值转成 1 通道（``convert_to_grayscale``）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# 1. 低层工具
# ---------------------------------------------------------------------------

def convert_to_grayscale(conv):
    """将3通道卷积层转换为单通道，权重取RGB均值。"""
    new_conv = nn.Conv2d(
        1, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups,
        bias=(conv.bias is not None), padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# 2. backbone
# ---------------------------------------------------------------------------

class ResNet18Stages(nn.Module):
    """resnet18 stem(stride 4) + layer1..4，输出 4 阶段特征 (stride 4/8/16/32)。"""

    channels = (64, 128, 256, 512)

    def __init__(self, pretrained):
        super().__init__()
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.conv1 = convert_to_grayscale(m.conv1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.stages = nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class ConvNeXtStages(nn.Module):
    """ConvNeXt features 8 个子模块（4 对 downsample+blocks）→ 输出 4 阶段特征。

    基类：``model_fn``（torchvision 的 ``convnext_*`` 构造函数）与
    ``stochastic_depth_prob``（随机深度正则强度，按深度线性增长、最深层取该值）
    由 :class:`ConvNeXtTinyStages` / :class:`ConvNeXtSmallStages` 子类提供。
    """

    channels = (96, 192, 384, 768)
    model_fn = None
    stochastic_depth_prob = None

    def __init__(self, pretrained):
        super().__init__()
        m = self.model_fn(
            weights="IMAGENET1K_V1" if pretrained else None,
            stochastic_depth_prob=self.stochastic_depth_prob,
        )
        m.features[0][0] = convert_to_grayscale(m.features[0][0])
        self.features = m.features

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i % 2 == 1:  # blocks 输出对应 stride 4/8/16/32
                feats.append(x)
        return feats


class ConvNeXtTinyStages(ConvNeXtStages):
    model_fn = staticmethod(torchvision.models.convnext_tiny)
    stochastic_depth_prob = 0.25


class ConvNeXtSmallStages(ConvNeXtStages):
    model_fn = staticmethod(torchvision.models.convnext_small)
    stochastic_depth_prob = 0.4


BACKBONES = {
    "resnet18": ResNet18Stages,
    "convnext_tiny": ConvNeXtTinyStages,
    "convnext_small": ConvNeXtSmallStages,
}


# ---------------------------------------------------------------------------
# 3. 检测头 + 顶层模型
# ---------------------------------------------------------------------------

class CenterHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.shared = nn.Sequential(ConvBNAct(ch, ch), ConvBNAct(ch, ch))
        self.hm_head = nn.Conv2d(ch, 1, 1)
        self.offset_head = nn.Conv2d(ch, 2, 1)
        # 初始 sigmoid(-2.19) ≈ 0.10：标准 CenterNet heatmap bias，
        # 降低早期背景响应，避免 focal loss 负样本项过大。
        nn.init.constant_(self.hm_head.bias, -2.19)

    def forward(self, x):
        x = self.shared(x)
        return {"hm": self.hm_head(x), "offset": self.offset_head(x)}


class CenterNet(nn.Module):
    """backbone → 4 阶段特征经 1×1 投影并双线性上采样到 stride-4 相加 → 两层卷积 → 中心头。"""

    def __init__(self, backbone="resnet18", pretrained=False, head_ch=128):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone 必须是 {list(BACKBONES)}，收到 {backbone!r}")
        if pretrained:
            print(f"Loading ImageNet pretrained weights for {backbone}")
        self.backbone = BACKBONES[backbone](pretrained=pretrained)
        self.proj = nn.ModuleList([nn.Conv2d(c, head_ch, 1, bias=False) for c in self.backbone.channels])
        self.fuse = nn.Sequential(ConvBNAct(head_ch, head_ch), ConvBNAct(head_ch, head_ch))
        self.head = CenterHead(head_ch)

    def forward(self, x):
        feats = self.backbone(x)
        target_size = feats[0].shape[-2:]
        fused = self.proj[0](feats[0])
        for proj, feat in zip(self.proj[1:], feats[1:]):
            fused = fused + F.interpolate(
                proj(feat), size=target_size, mode="bilinear", align_corners=False,
            )
        return self.head(self.fuse(fused))


def build_centernet_model(backbone="resnet18", pretrained=False, down_ratio=4, head_ch=128):
    if down_ratio != 4:
        raise ValueError(f"down_ratio 当前固定为 4（两类 backbone 的最浅特征均为 stride-4），收到 {down_ratio}")
    return CenterNet(backbone=backbone, pretrained=pretrained, head_ch=head_ch)


# ---------------------------------------------------------------------------
# 4. 损失
# ---------------------------------------------------------------------------

def _transpose_and_gather_feat(feat, ind):
    """从 ``[B, C, H, W]`` 的特征里按展平索引 ``ind`` (``[B, N]``) 取出 ``N`` 个位置 → ``[B, N, C]``。"""
    batch, channels, height, width = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(batch, height * width, channels)
    ind = ind.unsqueeze(2).expand(batch, ind.size(1), channels)
    return feat.gather(1, ind)


def focal_loss(logits, target, pos_weight=1.0, neg_weight=1.0):
    """CornerNet/CenterNet 的 penalty-reduced focal loss（heatmap 用），按正样本数归一。"""
    # AMP/FP16 下 1 - 1e-4 会被舍入为 1.0，导致 log(1 - pred) 变成 -inf。
    # heatmap focal loss 对数值稳定性敏感，因此强制在 FP32 中计算。
    logits = logits.float()
    target = target.float()
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    return -(
        pos_weight * pos_loss.sum() + neg_weight * neg_loss.sum()
    ) / torch.clamp(pos_inds.sum(), min=1.0)


def offset_l1_loss(pred_offset, target_offset, mask, ind):
    pred = _transpose_and_gather_feat(pred_offset, ind)
    mask = mask.unsqueeze(2)
    loss = F.l1_loss(pred * mask, target_offset * mask, reduction="sum")
    return loss / torch.clamp(mask.sum(), min=1.0)


def compute_loss(outputs, batch, hm_weight=1.0, offset_weight=1.0,
                 hm_pos_weight=1.0, hm_neg_weight=1.0):
    hm_loss = focal_loss(outputs["hm"], batch["hm"], hm_pos_weight, hm_neg_weight)
    off_loss = offset_l1_loss(outputs["offset"], batch["offset"], batch["reg_mask"], batch["ind"])
    loss = hm_weight * hm_loss + offset_weight * off_loss
    return loss, {"hm": hm_loss.detach(), "offset": off_loss.detach()}
