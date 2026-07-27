"""Binary burst classifier 模型：torchvision ConvNeXt（单通道 stem）。

* :class:`ConvNeXtNet`：标准 ConvNeXt + 2 类线性头；输入固定尺寸（默认 512²）。
* :class:`SPPConvNeXt`：ConvNeXt backbone + 空间金字塔池化，支持可变输入尺寸（配合 ``random_resize`` 多尺度训练）。

ImageNet 预训练 stem conv 的 3 通道权重会按通道取均值转成 1 通道（``convert_to_grayscale``）。
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms


# ---- 低层工具 ----
def convert_to_grayscale(conv):
    """将3通道卷积层转换为单通道，权重取RGB均值。"""
    new_conv = torch.nn.Conv2d(
        1, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups,
        bias=(conv.bias is not None), padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


# ---- backbone ----
# 模型构造函数 → 末层特征通道数
CONVNEXT_MODELS = {
    "convnext_tiny":  (torchvision.models.convnext_tiny,    768),
    "convnext_small": (torchvision.models.convnext_small,   768),
    "convnext_base":  (torchvision.models.convnext_base,   1024),
    "convnext_large": (torchvision.models.convnext_large,  1536),
}


def _build_convnext_base(model_name, pretrained):
    if model_name not in CONVNEXT_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Valid: {list(CONVNEXT_MODELS)}")
    model_fn, num_ch = CONVNEXT_MODELS[model_name]
    print(f"{'Loading ImageNet pretrained' if pretrained else 'Initializing'} {model_name}")
    basemodel = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
    basemodel.features[0][0] = convert_to_grayscale(basemodel.features[0][0])
    return basemodel, num_ch


# ---- 模型 ----
class ConvNeXtNet(torch.nn.Module):
    def __init__(self, model_name="convnext_tiny", num_classes=2, pretrained=False):
        super().__init__()
        self.base_model, num_ch = _build_convnext_base(model_name, pretrained)
        self.base_model.classifier[2] = torch.nn.Linear(num_ch, num_classes)

    def forward(self, x):
        return self.base_model(x)


class SpatialPyramidPool2D(torch.nn.Module):
    def __init__(self, out_side):
        super().__init__()
        self.pools = torch.nn.ModuleList(
            [torch.nn.AdaptiveMaxPool2d(output_size=(n, n)) for n in out_side]
        )

    def forward(self, x):
        return torch.cat([p(x).reshape(x.size(0), -1) for p in self.pools], dim=1)


class SPPConvNeXt(torch.nn.Module):
    def __init__(
        self,
        model_name="convnext_tiny",
        num_classes=2,
        pool_size=(1, 2, 6),
        dropout=0.5,
        pretrained=False,
    ):
        super().__init__()
        self.base_model, num_ch = _build_convnext_base(model_name, pretrained)
        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        num_features = num_ch * sum(n * n for n in pool_size)
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
        self.classifier = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.spp(x)
        x = self.dropout(x)
        return self.classifier(x)


def build_binary_model(
    model_type="ConvNeXtNet",
    model_name="convnext_tiny",
    num_classes=2,
    pretrained=False,
    dropout=0.5,
):
    if model_type == "ConvNeXtNet":
        return ConvNeXtNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    if model_type == "SPPConvNeXt":
        return SPPConvNeXt(
            model_name=model_name, num_classes=num_classes,
            dropout=dropout, pretrained=pretrained,
        )
    raise ValueError("model_type must be 'ConvNeXtNet' or 'SPPConvNeXt'")


# ---- 数据增强（多尺度训练用） ----
def random_resize(inputs):
    h, w = np.random.randint(128, 513), np.random.randint(128, 513)
    return torch.stack([transforms.Resize((h, w), antialias=True)(k) for k in inputs])
