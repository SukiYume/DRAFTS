"""
Radio Telescope Anomaly Detection - Autoencoder Models

- Encoder: ConvNeXt-Tiny (ImageNet pretrained), 1-channel input
- Decoder: Symmetric upsampling with gated skip connections, GroupNorm
- VAE: Convolutional latent space (1x1 conv)

Input:  (B, 1, 512, 512) in [0, 1]
Output: (B, 1, 512, 512) in [0, 1] (Sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import resnet18, ResNet18_Weights


# =============================================================================
# Loss Functions (using Huber/SmoothL1 for robustness)
# =============================================================================

def compute_ae_loss(recon, target, reduction='mean'):
    """
    Compute AE reconstruction loss (Huber/SmoothL1).

    Args:
        reduction: 'mean' for batch average, 'none' for per-sample

    Returns:
        (total_loss, recon_loss, kl_loss)  # kl_loss is always 0 for AE
    """
    if reduction == 'none':
        loss = F.smooth_l1_loss(recon, target, reduction='none').mean(dim=[1, 2, 3])
    else:
        loss = F.smooth_l1_loss(recon, target)
    return loss, loss, torch.tensor(0.0, device=recon.device)


def compute_vae_loss(recon, target, mu, logvar, beta=0.1, reduction='mean'):
    """
    Compute VAE loss = Huber + beta * KL.

    Args:
        recon: Reconstructed output
        target: Original input
        mu: Latent mean
        logvar: Latent log variance
        beta: KL weight
        reduction: 'mean' for batch average, 'none' for per-sample

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    if reduction == 'none':
        # Per-sample losses
        recon_loss = F.smooth_l1_loss(recon, target, reduction='none').mean(dim=[1, 2, 3])
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl_loss = kl.sum(dim=[1, 2, 3]) / kl[0].numel()
    else:
        # Batch average
        recon_loss = F.smooth_l1_loss(recon, target)
        kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# =============================================================================
# Encoder
# =============================================================================

class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny encoder with multi-scale features."""

    CHANNELS = [96, 192, 384, 768]

    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # Adapt stem for single-channel input
        orig_conv = backbone.features[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
            backbone.features[0][1],  # LayerNorm
        )

        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.stem[0].weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                self.stem[0].bias.copy_(orig_conv.bias)

        # Stages
        self.stage0 = backbone.features[1]   # 96ch, 128x128
        self.down1 = backbone.features[2]
        self.stage1 = backbone.features[3]   # 192ch, 64x64
        self.down2 = backbone.features[4]
        self.stage2 = backbone.features[5]   # 384ch, 32x32
        self.down3 = backbone.features[6]
        self.stage3 = backbone.features[7]   # 768ch, 16x16

    def forward(self, x):
        x = self.stem(x)
        f0 = self.stage0(x)
        f1 = self.stage1(self.down1(f0))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        return [f0, f1, f2, f3]


class ResNet18Encoder(nn.Module):
    """
    ResNet18 encoder with multi-scale features.

    与 ConvNeXtEncoder 结构类似，输出 4 个尺度的特征图：
    - f0: 128x128, 64ch
    - f1: 64x64, 128ch
    - f2: 32x32, 256ch
    - f3: 16x16, 512ch
    """

    CHANNELS = [64, 128, 256, 512]  # ResNet18 的通道数

    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt first conv for single-channel input
        # ResNet: conv1 (7x7, stride=2) -> bn1 -> relu -> maxpool (stride=2)
        # 输入 512 -> 经过 stem 后 128x128
        orig_conv = backbone.conv1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # 这里 stride=2，所以总 stride=4
        )

        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.stem[0].weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        # Stages (与 ConvNeXt 对应)
        self.stage0 = nn.Identity()       # 保持 64ch, 128x128
        self.stage1 = backbone.layer1     # 64ch -> 64ch, 保持尺寸 (但 ResNet18 layer1 不下采样)
        self.stage2 = backbone.layer2     # 64ch -> 128ch, 下采样 2x
        self.stage3 = backbone.layer3     # 128ch -> 256ch, 下采样 2x
        self.stage4 = backbone.layer4     # 256ch -> 512ch, 下采样 2x

        # 由于 ResNet 的 layer1 不下采样，需要额外处理
        # 为了与 ConvNeXtEncoder 兼容，我们调整输出：
        # f0: layer1 输出 (64ch, 128x128)
        # f1: layer2 输出 (128ch, 64x64)
        # f2: layer3 输出 (256ch, 32x32)
        # f3: layer4 输出 (512ch, 16x16)

    def forward(self, x):
        x = self.stem(x)           # -> (64, 128, 128)
        f0 = self.stage1(x)        # -> (64, 128, 128)
        f1 = self.stage2(f0)       # -> (128, 64, 64)
        f2 = self.stage3(f1)       # -> (256, 32, 32)
        f3 = self.stage4(f2)       # -> (512, 16, 16)
        return [f0, f1, f2, f3]


# =============================================================================
# Decoder
# =============================================================================

def _gn(ch, g=8):
    """GroupNorm with auto-adjusted groups."""
    g = min(g, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class DecoderBlock(nn.Module):
    """Upsample + skip concat + conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        mid = in_ch // 2

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, mid, 3, bias=False),
            _gn(mid),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid + skip_ch, out_ch, 3, bias=False),
            _gn(out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            _gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class Decoder(nn.Module):
    """
    Symmetric decoder with gated skip connections.

    Args:
        channels: [c0, c1, c2, c3] encoder 各层通道数
        out_channels: 输出通道数
    """

    GATE_INIT = -3.0

    def __init__(self, channels=None, out_channels=1):
        super().__init__()

        # 默认使用 ConvNeXt 通道配置
        if channels is None:
            channels = [96, 192, 384, 768]

        c0, c1, c2, c3 = channels

        self.up1 = DecoderBlock(c3, c2, c2)  # 16 -> 32
        self.up2 = DecoderBlock(c2, c1, c1)  # 32 -> 64

        # ===== 方案 A: 有 128x128 skip connection =====
        self.up3 = DecoderBlock(c1, c0, c0)    # 64 -> 128
        self.gates = nn.Parameter(torch.full((3,), self.GATE_INIT))

        # ===== 方案 B: 无 128x128 skip connection（当前使用）=====
        # self.up3 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(c1, c0, 3, bias=False),
        #     _gn(c0),
        #     nn.GELU(),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(c0, c0, 3, bias=False),
        #     _gn(c0),
        #     nn.GELU(),
        # )  # 64 -> 128
        # self.gates = nn.Parameter(torch.full((2,), self.GATE_INIT))

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c0, c0 // 2, 3, bias=False),
            _gn(c0 // 2),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c0 // 2, out_channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, features):
        f0, f1, f2, f3 = features
        g = F.softplus(self.gates)

        x = self.up1(f3, f2 * g[0])
        x = self.up2(x, f1 * g[1])

        # ===== 方案 A: 有 skip =====
        x = self.up3(x, f0 * g[2])

        # ===== 方案 B: 无 skip（当前使用）=====
        # x = self.up3(x)

        return self.final(x)


# 预定义的通道配置
ENCODER_CHANNELS = {
    'convnext': [96, 192, 384, 768],
    'resnet18': [64, 128, 256, 512],
}


# =============================================================================
# Autoencoder
# =============================================================================

class RadioAutoencoder(nn.Module):
    """
    Autoencoder for radio anomaly detection.

    Args:
        encoder_name: 'convnext' or 'resnet18'
    """

    def __init__(self, in_channels=1, pretrained=True, encoder_name='convnext'):
        super().__init__()

        self.encoder_name = encoder_name
        channels = ENCODER_CHANNELS[encoder_name]

        if encoder_name == 'resnet18':
            self.encoder = ResNet18Encoder(in_channels, pretrained)
        else:  # convnext (default)
            self.encoder = ConvNeXtEncoder(in_channels, pretrained)

        self.decoder = Decoder(channels=channels, out_channels=in_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        """Extract bottleneck features."""
        return self.encoder(x)[-1]

    @torch.no_grad()
    def get_anomaly_score(self, x):
        """Compute per-sample anomaly score (MSE)."""
        recon = self.forward(x)
        total, _, _ = compute_ae_loss(recon, x, reduction='none')
        return total


# =============================================================================
# Variational Autoencoder
# =============================================================================

class VariationalRadioAutoencoder(nn.Module):
    """
    Convolutional VAE for radio anomaly detection.

    Uses 1x1 conv for latent space (preserves spatial structure).

    Args:
        encoder_name: 'convnext' or 'resnet18'
    """

    def __init__(self, in_channels=1, pretrained=True, latent_ch=128, encoder_name='convnext'):
        super().__init__()

        self.encoder_name = encoder_name
        channels = ENCODER_CHANNELS[encoder_name]
        bottleneck_ch = channels[-1]  # 最后一层通道数

        if encoder_name == 'resnet18':
            self.encoder = ResNet18Encoder(in_channels, pretrained)
        else:  # convnext (default)
            self.encoder = ConvNeXtEncoder(in_channels, pretrained)

        self.decoder = Decoder(channels=channels, out_channels=in_channels)

        self.to_mu = nn.Conv2d(bottleneck_ch, latent_ch, 1)
        self.to_logvar = nn.Conv2d(bottleneck_ch, latent_ch, 1)
        self.from_z = nn.Conv2d(latent_ch, bottleneck_ch, 1)

        self.latent_ch = latent_ch

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        """Returns: (recon, mu, logvar)"""
        features = self.encoder(x)

        mu = self.to_mu(features[-1])
        logvar = self.to_logvar(features[-1])
        z = self.reparameterize(mu, logvar)

        features[-1] = self.from_z(z)
        return self.decoder(features), mu, logvar

    @torch.no_grad()
    def get_anomaly_score(self, x, beta=0.1):
        """
        Anomaly score = MSE + beta * KL (per sample).

        Using both reconstruction error and KL divergence allows detecting
        samples that have unusual latent representations.
        """
        recon, mu, logvar = self.forward(x)
        total, _, _ = compute_vae_loss(recon, x, mu, logvar, beta, reduction='none')
        return total

    @torch.no_grad()
    def encode(self, x, pool='gap'):
        """
        提取潜变量 mu（用于 UMAP/t-SNE 降维可视化）

        Args:
            x: [B, 1, 512, 512] 输入图像
            pool: 池化方式
                - 'gap': 全局平均池化 → [B, latent_ch]
                - 'flatten': 展平 → [B, latent_ch * 16 * 16]
                - None: 不处理 → [B, latent_ch, 16, 16]

        Returns:
            pool='gap':     [B, 128]
            pool='flatten': [B, 32768]
            pool=None:      [B, 128, 16, 16]
        """
        features = self.encoder(x)
        mu = self.to_mu(features[-1])  # [B, latent_ch, 16, 16]

        if pool == 'gap':
            return mu.mean(dim=[2, 3])  # [B, latent_ch]
        elif pool == 'flatten':
            return mu.flatten(1)  # [B, latent_ch * H * W]
        return mu


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(2, 1, 512, 512, device=device)

    print("=" * 60)
    print("Testing ConvNeXt AE...")
    ae = RadioAutoencoder(pretrained=False, encoder_name='convnext').to(device)
    y = ae(x)
    loss, recon, kl = compute_ae_loss(y, x)
    print(f"  Output: {y.shape}, Loss: {loss:.4f}")
    print(f"  Params: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f}M")

    print("\nTesting ResNet18 AE...")
    ae_r = RadioAutoencoder(pretrained=False, encoder_name='resnet18').to(device)
    y = ae_r(x)
    loss, recon, kl = compute_ae_loss(y, x)
    print(f"  Output: {y.shape}, Loss: {loss:.4f}")
    print(f"  Params: {sum(p.numel() for p in ae_r.parameters()) / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("Testing ConvNeXt VAE...")
    vae = VariationalRadioAutoencoder(pretrained=False, latent_ch=128, encoder_name='convnext').to(device)
    y, mu, logvar = vae(x)
    loss, recon, kl = compute_vae_loss(y, x, mu, logvar, beta=0.1)
    print(f"  Output: {y.shape}, Latent: {mu.shape}")
    print(f"  Loss: total={loss:.4f}, recon={recon:.4f}, kl={kl:.4f}")
    print(f"  Params: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")

    print("\nTesting ResNet18 VAE...")
    vae_r = VariationalRadioAutoencoder(pretrained=False, latent_ch=128, encoder_name='resnet18').to(device)
    y, mu, logvar = vae_r(x)
    loss, recon, kl = compute_vae_loss(y, x, mu, logvar, beta=0.1)
    print(f"  Output: {y.shape}, Latent: {mu.shape}")
    print(f"  Loss: total={loss:.4f}, recon={recon:.4f}, kl={kl:.4f}")
    print(f"  Params: {sum(p.numel() for p in vae_r.parameters()) / 1e6:.2f}M")
