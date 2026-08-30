# fmt: off

"""Autoencoder models and losses for radio-telescope anomaly detection.

Inputs and reconstructions use shape ``(B, 1, H, W)`` with values in [0, 1].
Both encoders downsample by 32, so H and W must be divisible by 32.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)


def compute_ae_loss(recon, target):
    """Return ``(total, reconstruction, kl)`` SmoothL1 losses for an AE."""
    loss = F.smooth_l1_loss(recon, target)
    return loss, loss, torch.zeros_like(loss)


def compute_vae_loss(recon, target, mu, logvar, beta=0.1):
    """Return ``(total, reconstruction, kl)`` losses for a beta-VAE."""
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    recon_loss = F.smooth_l1_loss(recon, target)
    kl_loss    = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def compute_anomaly_scores(recon, target, mu=None, logvar=None, beta=0.1):
    """Return total, reconstruction, and KL anomaly scores per sample.

    The reconstruction score is the larger of the mean and 99th-percentile
    pixelwise SmoothL1 errors. For a VAE, the mean KL divergence is added with
    weight ``beta``. AE callers should leave ``mu`` and ``logvar`` unset.
    """
    if (mu is None) != (logvar is None):
        raise ValueError(
            "mu and logvar must either both be provided or both be omitted"
        )
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    error       = F.smooth_l1_loss(recon, target, reduction="none").flatten(1)
    recon_score = torch.maximum(error.mean(dim=1), torch.quantile(error, 0.99, dim=1))

    if mu is None:
        kl_score = torch.zeros_like(recon_score)
    else:
        kl_score = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).flatten(1).mean(
            dim=1
        )

    return recon_score + beta * kl_score, recon_score, kl_score


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny encoder that exposes four feature resolutions."""

    CHANNELS = (96, 192, 384, 768)

    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        weights  = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)

        original_conv = backbone.features[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
            backbone.features[0][1],
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.stem[0].weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
                self.stem[0].bias.copy_(original_conv.bias)

        self.stage0 = backbone.features[1]
        self.down1  = backbone.features[2]
        self.stage1 = backbone.features[3]
        self.down2  = backbone.features[4]
        self.stage2 = backbone.features[5]
        self.down3  = backbone.features[6]
        self.stage3 = backbone.features[7]

    def forward(self, x):
        f0 = self.stage0(self.stem(x))
        f1 = self.stage1(self.down1(f0))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        return [f0, f1, f2, f3]


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder that exposes four feature resolutions."""

    CHANNELS = (64, 128, 256, 512)

    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        original_conv = backbone.conv1
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size = 7,
                stride      = 2,
                padding     = 3,
                bias        = False,
            ),
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.stem[0].weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        f0 = self.stage1(self.stem(x))
        f1 = self.stage2(f0)
        f2 = self.stage3(f1)
        f3 = self.stage4(f2)
        return [f0, f1, f2, f3]


def _group_norm(channels, groups=8):
    groups = min(groups, channels)
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DecoderBlock(nn.Module):
    """Upsample a feature map and merge one gated skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, mid_channels, 3, bias=False),
            _group_norm(mid_channels),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid_channels + skip_channels, out_channels, 3, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class Decoder(nn.Module):
    """Symmetric decoder with trainable gated skip connections."""

    GATE_INIT = -3.0

    def __init__(self, channels, out_channels=1):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.up1       = DecoderBlock(c3, c2, c2)
        self.up2       = DecoderBlock(c2, c1, c1)
        self.up3       = DecoderBlock(c1, c0, c0)
        self.gates     = nn.Parameter(torch.full((3,), self.GATE_INIT))
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c0, c0 // 2, 3, bias=False),
            _group_norm(c0 // 2),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c0 // 2, out_channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, features):
        f0, f1, f2, f3 = features
        gates          = F.softplus(self.gates)
        x              = self.up1(f3, f2 * gates[0])
        x              = self.up2(x, f1 * gates[1])
        x              = self.up3(x, f0 * gates[2])
        return self.final(x)


ENCODERS = {
    "convnext": ConvNeXtEncoder,
    "resnet18": ResNet18Encoder,
}


class RadioAutoencoder(nn.Module):
    """Convolutional autoencoder for reconstruction or next-slice prediction."""

    def __init__(self, in_channels=1, pretrained=True, encoder_name="convnext"):
        super().__init__()
        if encoder_name not in ENCODERS:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        self.encoder = ENCODERS[encoder_name](in_channels, pretrained)
        self.decoder = Decoder(self.encoder.CHANNELS, out_channels=in_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalRadioAutoencoder(nn.Module):
    """Convolutional VAE with a spatial 1x1-convolution latent layer."""

    def __init__(
        self,
        in_channels=1,
        pretrained=True,
        latent_ch=128,
        encoder_name="convnext",
    ):
        super().__init__()
        if encoder_name not in ENCODERS:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        if latent_ch <= 0:
            raise ValueError(f"latent_ch must be positive, got {latent_ch}")

        self.encoder        = ENCODERS[encoder_name](in_channels, pretrained)
        bottleneck_channels = self.encoder.CHANNELS[-1]
        self.decoder        = Decoder(self.encoder.CHANNELS, out_channels=in_channels)
        self.to_mu          = nn.Conv2d(bottleneck_channels, latent_ch, 1)
        self.to_logvar      = nn.Conv2d(bottleneck_channels, latent_ch, 1)
        self.from_z         = nn.Conv2d(latent_ch, bottleneck_channels, 1)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        """Return ``(reconstruction, mu, logvar)``.

        Training samples from the posterior. Evaluation decodes ``mu`` so
        anomaly scores and saved visualizations are deterministic.
        """
        features     = self.encoder(x)
        mu           = self.to_mu(features[-1])
        logvar       = self.to_logvar(features[-1])
        latent       = self.reparameterize(mu, logvar) if self.training else mu
        features[-1] = self.from_z(latent)
        return self.decoder(features), mu, logvar

# fmt: on
