"""
CT-MRI Fusion Training Pipeline — Updated Loss v3
===================================================
Architecture:
  - Frozen MAE ViT-Small encoders (CT + MRI), patch=16, embed=384, img=256
  - APFM fusion at every encoder layer (symmetric, layer-i fused → decoder layer-(N-i))
  - MAE ViT-Small decoder (trainable)

Loss v3 (Corrected from v2):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ ROOT CAUSE OF v2 FAILURE:                                           │
  │   max(ct,mri) target was unrealistic → model couldn't converge     │
  │   Laplacian loss had huge gradients → overwhelmed other signals     │
  │   λ_msssim=0.1 removed key structural regulariser                  │
  │   λ_freq=0.3, λ_percep=0.1 too aggressive on grayscale medical img │
  └─────────────────────────────────────────────────────────────────────┘

  Corrected Loss v3:
    L = λ_int    · L_int    (MAE vs mean target)          [1.0]  ← REVERTED
      + λ_ssim   · L_ssim   (1-SSIM vs mean target)       [0.5]
      + λ_msssim · L_msssim (1-MS-SSIM vs mean target)    [0.3]  ← restored
      + λ_grad   · L_grad   (Sobel, max-edge ref)          [0.5]
      + λ_texture· L_texture (local std deviation)         [0.3]  ← NEW (replaces Laplacian)
      + λ_freq   · L_freq   (FFT magnitude)                [0.15] ← halved
      + λ_percep · L_percep (VGG-16 relu2_2)              [0.05] ← halved
      + λ_modal  · L_modal  (dynamic modality)             [0.3]  ← restored

  vs Exp1 baseline:
    - L_int/L_ssim/L_msssim: still use mean target (stable, realistic)
    - λ_msssim: 0.5→0.3 (slight reduce, keeps structural regularisation)
    - L_texture (NEW, λ=0.3): local std deviation loss, directly attacks low SD/EN
    - L_freq    (NEW, λ=0.15): FFT magnitude, improves high-freq content gently
    - L_percep  (NEW, λ=0.05): VGG perceptual, gentle VIF improvement
    - λ_modal: restored to 0.3

Output layout:
  fusion_output_updatedLoss3/
  ├── best_model.pth
  ├── latest.pth
  ├── val_metrics.csv
  ├── final_metrics_summary.txt
  └── visualisations/
      └── epoch{NNN}/
          └── {patient_id}/          ← separate folder per patient
              └── img{NNNN}.png      ← 3-col grid: CT | MRI | Fused + metrics

Dataset layout (grayscale PNGs, paired by patient folder):
  /home/teaching/group46/MRI_dataset/{train,val}/{patient_id}/*.png
  /home/teaching/group46/CT_dataset/{train,val}/{patient_id}/*.png
"""

import os
import csv
import argparse
from pathlib import Path
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ssim, ms_ssim
from timm.models.vision_transformer import PatchEmbed, Block
from piq import vif_p


# ─────────────────────────────────────────────────────────────────────────────
# 1.  APFM  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class APFM(nn.Module):
    """Adaptive Pooling Fusion Module — fuses two feature maps of identical shape."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1   = nn.Conv2d(in_channels * 4, 2 * in_channels, 1)
        self.norm1   = nn.GroupNorm(min(32, in_channels), 2 * in_channels)
        self.silu    = nn.SiLU()
        self.conv2   = nn.Conv2d(2 * in_channels, in_channels, 1)
        self.norm2   = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv3   = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.norm3   = nn.GroupNorm(min(32, in_channels), in_channels)
        self.silu2   = nn.SiLU()
        self.conv4   = nn.Conv2d(in_channels, in_channels, 1)
        self.norm4   = nn.GroupNorm(min(32, in_channels), in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy     = torch.cat([x, y], dim=1)
        avg    = self.global_avg_pool(xy)
        mx     = self.global_max_pool(xy)
        pooled = torch.cat([avg, mx], dim=1)
        w_ch   = self.norm2(self.conv2(self.silu(self.norm1(self.conv1(pooled)))))
        w_sp   = self.norm4(self.conv4(self.silu2(self.norm3(self.conv3(xy)))))
        gate   = self.sigmoid(w_ch + w_sp)
        return x * gate + y * (1.0 - gate)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAE ViT-Small encoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int,
                             cls_token: bool = False) -> np.ndarray:
    """2-D sin-cos positional embedding (matches mae.py)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 4 == 0
    omega  = np.arange(embed_dim // 4, dtype=np.float32) / (embed_dim / 4)
    omega  = 1.0 / (10000 ** omega)

    def embed_1d(g, o):
        g   = g.reshape(-1)
        out = np.einsum('m,d->md', g, o)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = embed_1d(grid[0], omega)
    emb_w = embed_1d(grid[1], omega)
    emb   = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


class MAEEncoder(nn.Module):
    """
    MAE ViT-Small encoder — returns all intermediate layer outputs.
    img_size=256, patch=16 → 256 patches, embed_dim=384, depth=12, heads=12.
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=384, depth=12, num_heads=12, mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches      = self.patch_embed.num_patches
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list[depth] of (B, num_patches, embed_dim) tensors."""
        x   = self.patch_embed(x)
        x   = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(
            x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, 1:, :])       # strip cls → (B, 256, 384)
        x = self.norm(x)
        feats[-1] = x[:, 1:, :]             # replace last with normed
        return feats


def load_frozen_encoder(ckpt_path: str, device: torch.device) -> MAEEncoder:
    enc   = MAEEncoder()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    enc_keys = {k: v for k, v in state.items()
                if k.startswith(('patch_embed', 'cls_token',
                                  'pos_embed', 'blocks', 'norm'))}
    missing, unexpected = enc.load_state_dict(enc_keys, strict=False)
    print(f"[Encoder {Path(ckpt_path).name}]"
          f" missing={len(missing)}, unexpected={len(unexpected)}")
    enc.to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAE ViT-Small decoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class MAEDecoder(nn.Module):
    """
    ViT decoder — accepts per-layer fused tokens via symmetric skip connections.
    encoder depth=12 → decoder depth=8.
    """

    def __init__(self, num_patches=256, embed_dim=384,
                 decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, patch_size=16, in_chans=1,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.decoder_embed     = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token        = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans)
        self.skip_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self._init_weights(num_patches)

    def _init_weights(self, num_patches: int):
        pos = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, fused_tokens: torch.Tensor,
                skip_feats: List[torch.Tensor]) -> torch.Tensor:
        B, L, _ = fused_tokens.shape
        x       = self.decoder_embed(fused_tokens)
        cls_dec = torch.zeros(B, 1, x.shape[-1], device=x.device)
        x       = torch.cat([cls_dec, x], dim=1)
        x       = x + self.decoder_pos_embed

        for i, blk in enumerate(self.decoder_blocks):
            if i < len(skip_feats):
                skip = self.skip_proj(skip_feats[i])
                skip = torch.cat(
                    [torch.zeros(B, 1, skip.shape[-1], device=x.device),
                     skip], dim=1)
                x = x + skip
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])         # (B, L, p²·C)

        p = 16
        h = w = int(L ** .5)                        # 16
        x = x.reshape(B, h, w, p, p, 1)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, 1, h * p, w * p)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Fusion Model  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class CTMRIFusionModel(nn.Module):
    """
    Frozen CT-encoder + frozen MRI-encoder →
    per-layer APFM fusion (12 layers) →
    trainable MAE-ViT decoder with symmetric skips.
    """
    ENCODER_DEPTH = 12
    DECODER_DEPTH = 8

    def __init__(self, ct_ckpt: str, mri_ckpt: str, device: torch.device):
        super().__init__()
        self.ct_enc  = load_frozen_encoder(ct_ckpt,  device)
        self.mri_enc = load_frozen_encoder(mri_ckpt, device)

        embed_dim = 384
        self.apfm_layers = nn.ModuleList([
            APFM(embed_dim) for _ in range(self.ENCODER_DEPTH)
        ])
        self.decoder = MAEDecoder(
            num_patches=256, embed_dim=embed_dim,
            decoder_embed_dim=512, decoder_depth=self.DECODER_DEPTH,
            decoder_num_heads=16, patch_size=16, in_chans=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, 256, 384) → (B, 384, 16, 16)"""
        B, L, C = tokens.shape
        g = int(L ** .5)
        return tokens.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()

    def _spatial_to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, 384, 16, 16) → (B, 256, 384)"""
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    def forward(self, ct: torch.Tensor, mri: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ct_feats  = self.ct_enc(ct)
            mri_feats = self.mri_enc(mri)

        fused_layers = []
        for i, apfm in enumerate(self.apfm_layers):
            ct_s  = self._tokens_to_spatial(ct_feats[i])
            mri_s = self._tokens_to_spatial(mri_feats[i])
            fused_layers.append(self._spatial_to_tokens(apfm(ct_s, mri_s)))

        final_fused = fused_layers[self.ENCODER_DEPTH - 1]
        skip_feats  = [fused_layers[self.ENCODER_DEPTH - 2 - i]
                       for i in range(self.DECODER_DEPTH)]
        return self.decoder(final_fused, skip_feats)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Dataset  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """Loads paired grayscale CT & MRI PNGs, normalised to [0, 1]."""

    def __init__(self, ct_root: str, mri_root: str, split: str = 'train'):
        self.ct_root  = Path(ct_root)  / split
        self.mri_root = Path(mri_root) / split
        self.pairs: List[Tuple[Path, Path, str]] = []

        ct_pats  = sorted(p.name for p in self.ct_root.iterdir()
                          if p.is_dir())
        mri_pats = sorted(p.name for p in self.mri_root.iterdir()
                          if p.is_dir())
        shared   = sorted(set(ct_pats) & set(mri_pats))

        for pid in shared:
            ct_slices  = sorted((self.ct_root  / pid).glob('*.png'))
            mri_slices = sorted((self.mri_root / pid).glob('*.png'))
            n = min(len(ct_slices), len(mri_slices))
            for i in range(n):
                self.pairs.append((ct_slices[i], mri_slices[i], pid))

        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),          # → [0, 1]
        ])
        print(f"[Dataset/{split}] {len(self.pairs)} pairs"
              f" from {len(shared)} patients")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_p, mri_p, pid = self.pairs[idx]
        return (self.tf(Image.open(ct_p)),
                self.tf(Image.open(mri_p)),
                pid)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CORRECTED LOSS FUNCTIONS  (v3)
# ─────────────────────────────────────────────────────────────────────────────

class GradientLoss(nn.Module):
    """Sobel first-order gradient loss (unchanged from Exp1)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0,-1],[2, 0,-2],[1, 0,-1]],
                           dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1, 2, 1],[0, 0, 0],[-1,-2,-1]],
                           dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, ref_ct, ref_mri):
        def grad_mag(x):
            gx = F.conv2d(x, self.kx, padding=1)
            gy = F.conv2d(x, self.ky, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)

        gp    = grad_mag(pred)
        g_ref = torch.max(grad_mag(ref_ct), grad_mag(ref_mri))
        return F.l1_loss(gp, g_ref)


# ── NEW (replaces failed Laplacian) ─────────────────────────────────────────
class LocalTextureLoss(nn.Module):
    """
    Local Standard Deviation loss.

    Computes per-pixel local std via a sliding 5×5 window and penalises
    differences between pred and target in local texture richness.

    Why this instead of Laplacian:
      - Laplacian responses have large magnitudes that overwhelm gradients
      - Local std is bounded, numerically stable, and directly measures
        the spatial variation (SD) we want to increase
      - Minimising this loss forces the model to reproduce local texture
        density, directly fixing the low SD/EN observed in Exp1

    λ = 0.3
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.ks  = kernel_size
        self.pad = kernel_size // 2

    def _local_std(self, x: torch.Tensor) -> torch.Tensor:
        mu  = F.avg_pool2d(x, self.ks, stride=1, padding=self.pad)
        mu2 = F.avg_pool2d(x ** 2, self.ks, stride=1, padding=self.pad)
        var = (mu2 - mu ** 2).clamp(min=0)
        return var.sqrt()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._local_std(pred), self._local_std(target))


# ── NEW (halved weight vs v2) ────────────────────────────────────────────────
class FFTLoss(nn.Module):
    """
    2-D FFT magnitude spectrum loss.
    Gently encourages high-frequency content reproduction.
    λ = 0.15 (halved from v2's 0.3 to avoid destabilising convergence).
    """

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.rfft2(pred,   norm='ortho')
        fft_tgt  = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(fft_pred.abs(), fft_tgt.abs())


# ── NEW (halved weight vs v2) ────────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    """
    VGG-16 relu2_2 feature-space loss (pretrained ImageNet, fully frozen).
    Grayscale → 3-channel repeat → ImageNet normalise → VGG features.
    λ = 0.05 (halved from v2's 0.1; VGG is trained on colour images,
    so a gentle weight avoids misleading gradients on grayscale medical data).
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_feat = nn.Sequential(*list(vgg.children())[:9])
        for p in self.vgg_feat.parameters():
            p.requires_grad_(False)
        self.vgg_feat.eval()

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.vgg_feat(self._preprocess(pred)),
                         self.vgg_feat(self._preprocess(target)))


class DynamicModalityLoss(nn.Module):
    """
    Dynamic modality weighting (unchanged from Exp1).
    λ restored to 0.3.
    """

    def forward(self, pred, ct, mri):
        def local_var(x):
            mu  = F.avg_pool2d(x, 7, stride=1, padding=3)
            mu2 = F.avg_pool2d(x ** 2, 7, stride=1, padding=3)
            return (mu2 - mu ** 2).clamp(min=0)

        var_ct  = local_var(ct)
        var_mri = local_var(mri)
        total   = var_ct + var_mri + 1e-8
        w_ct    = var_ct  / total
        w_mri   = var_mri / total
        return (w_ct * (pred - ct) ** 2 +
                w_mri * (pred - mri) ** 2).mean()


# ── CORRECTED FUSION LOSS v3 ─────────────────────────────────────────────────
class FusionLoss(nn.Module):
    """
    Corrected Fusion Loss v3.

    Key corrections from v2:
      1. target = 0.5*(ct+mri)  — REVERTED from max(ct,mri)
         max-target was unrealistic and caused large, unlearnable residuals
      2. L_lap REMOVED — Laplacian gradients were numerically too large
      3. L_texture ADDED (local std, λ=0.3) — stable, bounded texture loss
      4. λ_msssim restored to 0.3 (was cut too aggressively to 0.1 in v2)
      5. λ_freq halved to 0.15  (was 0.3 in v2)
      6. λ_percep halved to 0.05 (was 0.1 in v2)
      7. λ_modal restored to 0.3 (was 0.2 in v2)

    Component summary:
      L_int     λ=1.0   MAE vs 0.5*(ct+mri)            ← stable regression target
      L_ssim    λ=0.5   1-SSIM vs mean target
      L_msssim  λ=0.3   1-MS-SSIM vs mean target        ← restored from 0.1
      L_grad    λ=0.5   Sobel, max-edge ref              ← unchanged
      L_texture λ=0.3   Local std deviation (NEW)        ← fixes low SD/EN
      L_freq    λ=0.15  FFT magnitude (NEW, gentler)     ← fixes high-freq
      L_percep  λ=0.05  VGG-16 relu2_2 (NEW, gentler)   ← improves VIF
      L_modal   λ=0.3   Dynamic modality                 ← restored from 0.2
    """

    def __init__(self,
                 lambda_int     = 1.0,
                 lambda_ssim    = 0.5,
                 lambda_msssim  = 0.3,
                 lambda_grad    = 0.5,
                 lambda_texture = 0.3,
                 lambda_freq    = 0.15,
                 lambda_percep  = 0.05,
                 lambda_modal   = 0.3):
        super().__init__()
        self.lambda_int     = lambda_int
        self.lambda_ssim    = lambda_ssim
        self.lambda_msssim  = lambda_msssim
        self.lambda_grad    = lambda_grad
        self.lambda_texture = lambda_texture
        self.lambda_freq    = lambda_freq
        self.lambda_percep  = lambda_percep
        self.lambda_modal   = lambda_modal

        self.grad_loss    = GradientLoss()
        self.texture_loss = LocalTextureLoss(kernel_size=5)
        self.fft_loss     = FFTLoss()
        self.percep_loss  = PerceptualLoss()
        self.modal_loss   = DynamicModalityLoss()

    def forward(self, pred: torch.Tensor,
                ct: torch.Tensor,
                mri: torch.Tensor):
        """All tensors: (B, 1, 256, 256) in [0, 1]."""

        # ── Stable mean target (REVERTED from max in v2) ────────────────────
        target = 0.5 * (ct + mri)

        # L_int — pixel MAE
        l_int = F.l1_loss(pred, target)

        # L_ssim — structural similarity
        l_ssim = 1.0 - ssim(pred, target,
                             data_range=1.0, size_average=True)

        # L_msssim — multi-scale SSIM  (λ restored: 0.1 → 0.3)
        l_msssim = 1.0 - ms_ssim(pred, target,
                                  data_range=1.0, size_average=True)

        # L_grad — Sobel sharpest-edge reference
        l_grad = self.grad_loss(pred, ct, mri)

        # L_texture — local std deviation  (NEW, replaces Laplacian)
        l_texture = self.texture_loss(pred, target)

        # L_freq — FFT magnitude spectrum  (NEW, λ halved to 0.15)
        l_freq = self.fft_loss(pred, target)

        # L_percep — VGG-16 perceptual  (NEW, λ halved to 0.05)
        l_percep = self.percep_loss(pred, target)

        # L_modal — dynamic modality  (λ restored: 0.2 → 0.3)
        l_modal = self.modal_loss(pred, ct, mri)

        total = (self.lambda_int     * l_int     +
                 self.lambda_ssim    * l_ssim    +
                 self.lambda_msssim  * l_msssim  +
                 self.lambda_grad    * l_grad    +
                 self.lambda_texture * l_texture +
                 self.lambda_freq    * l_freq    +
                 self.lambda_percep  * l_percep  +
                 self.lambda_modal   * l_modal)

        components = dict(
            int     = l_int.item(),
            ssim    = l_ssim.item(),
            msssim  = l_msssim.item(),
            grad    = l_grad.item(),
            texture = l_texture.item(),
            freq    = l_freq.item(),
            percep  = l_percep.item(),
            modal   = l_modal.item(),
        )
        return total, components


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation Metrics  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred: torch.Tensor,
                    ct:   torch.Tensor,
                    mri:  torch.Tensor) -> dict:
    """
    All tensors: (B, 1, H, W) in [0,1] on device.
    Reference = mean of CT and MRI.
    Returns per-batch averaged metrics.
    """
    ref = 0.5 * (ct + mri)

    def psnr(a, b):
        mse = ((a - b) ** 2).mean(dim=[1, 2, 3])
        return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()

    p = pred.clamp(0, 1)
    r = ref.clamp(0, 1)

    psnr_v   = psnr(p, r)
    ssim_v   = ssim(p, r,    data_range=1.0, size_average=True).item()
    msssim_v = ms_ssim(p, r, data_range=1.0, size_average=True).item()
    vif_v    = vif_p(p, r,   data_range=1.0).item()
    mse_v    = F.mse_loss(p, r).item()
    mae_v    = F.l1_loss(p, r).item()

    pred_np = p.squeeze(1).cpu().numpy()          # (B, H, W)
    sd_v    = float(pred_np.std())
    hist, _ = np.histogram(pred_np, bins=256, range=(0, 1), density=True)
    hist    = hist + 1e-10
    hist   /= hist.sum()
    en_v    = float(-(hist * np.log2(hist)).sum())

    overall = (psnr_v / 50.0 + ssim_v + msssim_v + vif_v) / 4.0

    return dict(psnr=psnr_v, ssim=ssim_v, ms_ssim=msssim_v,
                vif=vif_v, sd=sd_v, en=en_v,
                mse=mse_v, mae=mae_v, overall=overall)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Visualisation — 3-column grid (CT | MRI | Fused) + per-image metrics
#     Saved to: vis_dir / epoch{NNN} / {patient_id} / img{NNNN}.png
# ─────────────────────────────────────────────────────────────────────────────
def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, H, W) float [0,1] → 8-bit PIL grayscale image."""
    arr = t.squeeze(0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def save_fusion_grid(ct:      torch.Tensor,
                     mri:     torch.Tensor,
                     pred:    torch.Tensor,
                     metrics: dict,
                     save_path: Path) -> None:
    """
    Save a 3-column PNG:
        Col 0 — CT  (input)
        Col 1 — MRI (input)
        Col 2 — Fused image + all 9 evaluation metrics printed below

    Parameters
    ----------
    ct, mri, pred : single-image tensors  (1, H, W)  in [0, 1]
    metrics       : dict from compute_metrics() for this single image
    save_path     : full output path (parent dirs created automatically)
    """
    IMG_W, IMG_H = 256, 256
    PAD          = 10
    TITLE_H      = 30
    METRIC_H     = 120
    COLS         = 3

    canvas_w = COLS * IMG_W + (COLS + 1) * PAD
    canvas_h = TITLE_H + IMG_H + METRIC_H + 2 * PAD

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(235, 235, 242))
    draw   = ImageDraw.Draw(canvas)

    # ── Fonts ────────────────────────────────────────────────────────────────
    FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    FONT_PATHS_REG = [p.replace("Bold", "").replace("-Bold", "")
                      for p in FONT_PATHS]

    font_title = font_metric = None
    for fp, fr in zip(FONT_PATHS, FONT_PATHS_REG):
        try:
            font_title  = ImageFont.truetype(fp, 15)
            font_metric = ImageFont.truetype(fr, 11)
            break
        except Exception:
            continue
    if font_title is None:
        font_title = font_metric = ImageFont.load_default()

    # ── Column headers ────────────────────────────────────────────────────────
    col_titles = ['CT  (input)', 'MRI  (input)', 'Fused  (output)']
    col_images = [_tensor_to_pil(ct),
                  _tensor_to_pil(mri),
                  _tensor_to_pil(pred)]
    title_colours = [(60, 60, 60), (60, 60, 60), (10, 60, 130)]

    for col_idx, (title, img, tc) in enumerate(
            zip(col_titles, col_images, title_colours)):

        x0    = PAD + col_idx * (IMG_W + PAD)
        y_img = TITLE_H + PAD

        # Title
        draw.text((x0 + 4, 7), title, fill=tc, font=font_title)

        # Image
        canvas.paste(img.convert('RGB'), (x0, y_img))

        # Separator line
        y_sep = y_img + IMG_H + 4
        draw.line([(x0, y_sep), (x0 + IMG_W, y_sep)],
                  fill=(160, 160, 170), width=1)

        # Metrics text — Fused column only
        y_text = y_sep + 6
        if col_idx == 2:
            lines = [
                f"PSNR    : {metrics['psnr']:.2f} dB",
                f"SSIM    : {metrics['ssim']:.4f}",
                f"MS-SSIM : {metrics['ms_ssim']:.4f}",
                f"VIF     : {metrics['vif']:.4f}",
                f"SD      : {metrics['sd']:.4f}",
                f"EN      : {metrics['en']:.4f}",
                f"MSE     : {metrics['mse']:.5f}",
                f"MAE     : {metrics['mae']:.5f}",
                f"OVERALL : {metrics['overall']:.4f}",
            ]
            for li, line in enumerate(lines):
                draw.text((x0 + 5, y_text + li * 12),
                          line, fill=(5, 5, 110), font=font_metric)
        else:
            draw.text((x0 + 5, y_text + 50),
                      "metrics shown →", fill=(130, 130, 130),
                      font=font_metric)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(save_path))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for step, (ct, mri, _) in enumerate(loader):
        ct, mri = ct.to(device), mri.to(device)
        pred    = model(ct, mri)
        loss, comps = criterion(pred, ct, mri)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % 50 == 0:
            comp_str = "  ".join(
                f"{k}={v:.4f}" for k, v in comps.items())
            print(f"  [E{epoch} {step}/{len(loader)}]"
                  f" loss={loss.item():.4f}  {comp_str}")

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Validation Loop
#     Per-image grids saved to:
#       vis_dir / epoch{NNN} / {patient_id} / img{NNNN}.png
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, epoch,
             csv_path: str, best_psnr: float, vis_dir: Path):
    model.eval()
    rows     = []
    val_loss = 0.0
    agg      = {k: 0.0 for k in
                ('psnr', 'ssim', 'ms_ssim', 'vif',
                 'sd', 'en', 'mse', 'mae', 'overall')}

    # Per-patient image counter — keeps filename indices unique per patient
    patient_img_counters: dict = {}

    for ct, mri, pids in loader:
        ct, mri = ct.to(device), mri.to(device)
        pred    = model(ct, mri)
        loss, _ = criterion(pred, ct, mri)
        val_loss += loss.item()

        m = compute_metrics(pred, ct, mri)
        for k in agg:
            agg[k] += m[k]

        for i, pid in enumerate(pids):
            # Per-image metrics
            pm = compute_metrics(
                pred[i:i+1], ct[i:i+1], mri[i:i+1])
            rows.append({'patient': pid, 'epoch': epoch, **pm})

            # ── Save visualisation grid ──────────────────────────────────
            # Path: vis_dir / epoch{NNN} / {patient_id} / img{NNNN}.png
            if pid not in patient_img_counters:
                patient_img_counters[pid] = 0

            img_idx = patient_img_counters[pid]
            patient_img_counters[pid] += 1

            grid_path = (vis_dir
                         / f"epoch{epoch:03d}"
                         / pid                        # ← per-patient folder
                         / f"img{img_idx:04d}.png")

            save_fusion_grid(
                ct[i],
                mri[i],
                pred[i].clamp(0, 1),
                pm,
                grid_path,
            )

    n = len(loader)
    for k in agg:
        agg[k] /= n

    # ── Append per-image rows to CSV ─────────────────────────────────────────
    fieldnames = ['epoch', 'patient',
                  'psnr', 'ssim', 'ms_ssim', 'vif',
                  'sd', 'en', 'mse', 'mae', 'overall']
    write_header = not Path(csv_path).exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{row[k]:.6f}"
                     if k not in ('patient', 'epoch')
                     else row[k])
                 for k in fieldnames})

    avg_psnr = agg['psnr']
    print(f"  [Val E{epoch}] loss={val_loss/n:.4f}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    return val_loss / n, avg_psnr


# ─────────────────────────────────────────────────────────────────────────────
# 11. Early Stopping  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# 12. Final Evaluation  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def final_evaluation(model, loader, device):
    model.eval()
    agg   = {k: 0.0 for k in
             ('psnr', 'ssim', 'ms_ssim', 'vif',
              'sd', 'en', 'mse', 'mae', 'overall')}
    count = 0
    for ct, mri, _ in loader:
        ct, mri = ct.to(device), mri.to(device)
        pred    = model(ct, mri)
        m       = compute_metrics(pred, ct, mri)
        for k in agg:
            agg[k] += m[k]
        count += 1
    for k in agg:
        agg[k] /= count

    print("\n===== FINAL EVALUATION (Corrected Loss v3) =====")
    for k, v in agg.items():
        print(f"  {k.upper():10s}: {v:.4f}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 13. Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='CT-MRI Fusion — Corrected Loss v3')
    p.add_argument('--ct_root',
                   default='/home/teaching/group46/CT_dataset')
    p.add_argument('--mri_root',
                   default='/home/teaching/group46/MRI_dataset')
    p.add_argument('--ct_ckpt',
                   default='/home/teaching/group46/MAEDiff-main/'
                           'mae_vit_small_patch16_ct/encoder.pth')
    p.add_argument('--mri_ckpt',
                   default='/home/teaching/group46/MAEDiff-main/'
                           'mae_vit_small_patch16_mri/encoder.pth')
    # ── Output folder: fusion_output_updatedLoss3 ────────────────────────────
    p.add_argument('--out_dir',
                   default='/home/teaching/group46/MAEDiff-main/'
                           'fusion_output_updatedLoss3')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--resume',      default='')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir  = out_dir / 'visualisations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(out_dir / 'val_metrics.csv')

    print(f"Out dir : {out_dir}")
    print(f"Visuals : {vis_dir}  (layout: epoch{{N}}/{{patient}}/img{{N}}.png)")
    print(f"CSV     : {csv_path}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = FusionDataset(args.ct_root, args.mri_root, 'train')
    val_ds   = FusionDataset(args.ct_root, args.mri_root, 'val')
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = CTMRIFusionModel(
        args.ct_ckpt, args.mri_ckpt, device).to(device)
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters()
                    if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Corrected Loss v3 ─────────────────────────────────────────────────────
    criterion = FusionLoss(
        lambda_int     = 1.0,
        lambda_ssim    = 0.5,
        lambda_msssim  = 0.3,
        lambda_grad    = 0.5,
        lambda_texture = 0.3,
        lambda_freq    = 0.15,
        lambda_percep  = 0.05,
        lambda_modal   = 0.3,
    ).to(device)

    print("\nLoss config (Corrected v3):")
    print("  L_int     λ=1.00  MAE vs 0.5*(ct+mri)     [REVERTED from max]")
    print("  L_ssim    λ=0.50  1-SSIM vs mean target")
    print("  L_msssim  λ=0.30  1-MS-SSIM vs mean target [restored from 0.1]")
    print("  L_grad    λ=0.50  Sobel gradient            [unchanged]")
    print("  L_texture λ=0.30  Local std deviation       [NEW, replaces Laplacian]")
    print("  L_freq    λ=0.15  FFT magnitude spectrum    [NEW, halved from 0.3]")
    print("  L_percep  λ=0.05  VGG-16 relu2_2            [NEW, halved from 0.1]")
    print("  L_modal   λ=0.30  Dynamic modality          [restored from 0.2]")

    start_epoch = 1
    best_psnr   = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -1.0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(patience=10)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}"
              f"   lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, device, epoch)

        val_loss, avg_psnr = validate(
            model, val_dl, criterion, device, epoch,
            csv_path, best_psnr, vis_dir)

        scheduler.step()

        # Save latest checkpoint
        ckpt_data = dict(
            epoch      = epoch,
            model      = model.state_dict(),
            optimizer  = optimizer.state_dict(),
            scheduler  = scheduler.state_dict(),
            best_psnr  = best_psnr,
            train_loss = train_loss,
            val_loss   = val_loss,
        )
        torch.save(ckpt_data, out_dir / 'latest.pth')

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(ckpt_data, out_dir / 'best_model.pth')
            print(f"  ★ Best model saved  (PSNR={best_psnr:.4f})")

        if early_stopper.step(avg_psnr):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining complete.  Best PSNR : {best_psnr:.4f}")
    print(f"CSV          : {csv_path}")
    print(f"Checkpoints  : {out_dir}/")
    print(f"Visual grids : {vis_dir}/")

    # ── Final Evaluation on best model ────────────────────────────────────────
    print("\nLoading best model for final evaluation …")
    best_ckpt = torch.load(out_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_ckpt['model'])
    final_metrics = final_evaluation(model, val_dl, device)

    # Write plain-text summary
    summary_path = out_dir / 'final_metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("===== FINAL EVALUATION — Corrected Loss v3 =====\n\n")
        f.write("Loss Config (corrections from v2):\n")
        f.write("  L_int     λ=1.00  MAE vs 0.5*(ct+mri)     ← REVERTED from max\n")
        f.write("  L_ssim    λ=0.50  1-SSIM vs mean target\n")
        f.write("  L_msssim  λ=0.30  1-MS-SSIM               ← restored from 0.1\n")
        f.write("  L_grad    λ=0.50  Sobel gradient           ← unchanged\n")
        f.write("  L_texture λ=0.30  Local std deviation      ← NEW (replaces Laplacian)\n")
        f.write("  L_freq    λ=0.15  FFT magnitude            ← NEW, halved from 0.3\n")
        f.write("  L_percep  λ=0.05  VGG-16 relu2_2           ← NEW, halved from 0.1\n")
        f.write("  L_modal   λ=0.30  Dynamic modality         ← restored from 0.2\n\n")
        f.write("Metrics:\n")
        for k, v in final_metrics.items():
            f.write(f"  {k.upper():10s}: {v:.4f}\n")

    print(f"Summary : {summary_path}")


if __name__ == '__main__':
    main()