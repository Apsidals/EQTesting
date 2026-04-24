"""Rung 5: Split architecture — Universal Physics Encoder + Regional Site
Encoder + Fusion head. This is the hypothesis.

Hard contracts (enforced by tests/test_physics_branch_has_no_regional_input.py):
- `UniversalPhysicsEncoder.forward` takes ONLY a waveform tensor. No lat/lon,
  no Vs30, no region label, no station id. Ever.
- `RegionalSiteEncoder.forward` takes ONLY site features — never the raw
  waveform.
- `SplitTransferModel.forward(waveform, site_feats)` is the single entry
  point that composes them.

No BatchNorm anywhere (CLAUDE.md: BatchNorm destroys amplitude scale).
No attention yet (CLAUDE.md: add only after an ablation shows it wins).
No amplitude normalization inside the model (magnitude IS amplitude).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig
from ..data.geological import SITE_FEATURE_DIM


def _groupnorm(groups: int, channels: int) -> nn.GroupNorm:
    # GroupNorm's contract: num_channels must be divisible by num_groups.
    # Back off to 1 group (= LayerNorm-ish) if the user's hparams don't divide.
    g = groups if channels % groups == 0 else 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        groupnorm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm = _groupnorm(groupnorm_groups, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class UniversalPhysicsEncoder(nn.Module):
    """1D CNN over 3-component waveforms → physics embedding.

    Forward signature is (waveform,) ONLY. This is a hard invariant of the
    whole paper; if you extend the signature, the transfer claim collapses."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_dim = cfg.phys_embed_dim

        blocks: list[nn.Module] = []
        in_ch = 3  # Z, N, E
        for out_ch in cfg.phys_channels:
            blocks.append(
                ConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=cfg.phys_kernel_size,
                    stride=cfg.phys_stride,
                    groupnorm_groups=cfg.phys_groupnorm_groups,
                    dropout=cfg.phys_dropout,
                )
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, cfg.phys_embed_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 3:
            raise ValueError(
                f"UniversalPhysicsEncoder expects (B, 3, W), got {tuple(waveform.shape)}"
            )
        x = self.conv(waveform)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.head(x)


class RegionalSiteEncoder(nn.Module):
    """MLP over site features (Vs30, crustal Vp, sediment thickness, NEHRP
    one-hot, instrument one-hot). Never sees the waveform."""

    def __init__(self, cfg: ModelConfig, input_dim: int = SITE_FEATURE_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = cfg.site_embed_dim
        dims = [input_dim, *cfg.site_hidden, cfg.site_embed_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(cfg.site_dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, site_feats: torch.Tensor) -> torch.Tensor:
        if site_feats.ndim != 2 or site_feats.shape[1] != self.input_dim:
            raise ValueError(
                f"RegionalSiteEncoder expects (B, {self.input_dim}), got "
                f"{tuple(site_feats.shape)}"
            )
        return self.net(site_feats)


class FusionHead(nn.Module):
    """Concat(physics_embed, site_embed) → MLP → scalar magnitude."""

    def __init__(self, cfg: ModelConfig, phys_dim: int, site_dim: int) -> None:
        super().__init__()
        dims = [phys_dim + site_dim, *cfg.fusion_hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(cfg.fusion_dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, phys_embed: torch.Tensor, site_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([phys_embed, site_embed], dim=-1)
        return self.net(fused).squeeze(-1)


@dataclass
class SplitForward:
    prediction: torch.Tensor
    physics_embed: torch.Tensor
    site_embed: torch.Tensor


class SplitTransferModel(nn.Module):
    """Composes UniversalPhysicsEncoder + RegionalSiteEncoder + FusionHead.

    Training (rung 5 in-region): full model trained end-to-end on California.
    Zero-shot transfer: forward(waveform, site_feats) on target region, no
        parameter changes.
    Few-shot transfer: freeze the universal encoder, re-fit site encoder +
        fusion on small amount of target data.
    """

    def __init__(self, cfg: ModelConfig, site_feature_dim: int = SITE_FEATURE_DIM) -> None:
        super().__init__()
        self.cfg = cfg
        self.universal = UniversalPhysicsEncoder(cfg)
        self.site = RegionalSiteEncoder(cfg, input_dim=site_feature_dim)
        self.fusion = FusionHead(
            cfg, phys_dim=self.universal.embed_dim, site_dim=self.site.embed_dim
        )

    def forward(
        self,
        waveform: torch.Tensor,
        site_feats: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | SplitForward:
        phys = self.universal(waveform)
        site = self.site(site_feats)
        y = self.fusion(phys, site)
        if return_embeddings:
            return SplitForward(prediction=y, physics_embed=phys, site_embed=site)
        return y

    def encode_physics(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.universal(waveform)

    def encode_site(self, site_feats: torch.Tensor) -> torch.Tensor:
        return self.site(site_feats)

    def freeze_universal(self) -> None:
        """Freeze universal encoder params — called before few-shot transfer.
        Site encoder and fusion head remain trainable."""
        for p in self.universal.parameters():
            p.requires_grad = False
        self.universal.eval()

    def unfreeze_universal(self) -> None:
        for p in self.universal.parameters():
            p.requires_grad = True
        self.universal.train()

    def reset_site_and_fusion(self) -> None:
        """Reinitialize site encoder + fusion weights. Used when few-shot
        adapting to a new target region: the universal encoder carries
        pretrained source knowledge forward; the site/fusion weights
        start fresh on target data."""

        def _reset(m: nn.Module) -> None:
            for child in m.modules():
                if isinstance(child, (nn.Linear, nn.Conv1d)):
                    child.reset_parameters()

        _reset(self.site)
        _reset(self.fusion)
