"""
CT-MRI Fusion — Combined Training + Inference Pipeline
=======================================================
Architecture:
  - Frozen MAE ViT-Small encoders (CT + MRI), patch=16, embed=384, img=256
  - APFM fusion at every encoder layer (symmetric, layer-i fused → decoder layer-(N-i))
  - MAE ViT-Small decoder (trainable)
  - Fixed Balanced Fusion Loss (DynamicModalityLoss bias corrected)

Dataset layout (grayscale PNGs, paired by patient folder):
  Training/Validation:
    /home/teaching/group46/MRI_dataset/{train,val}/{patient_id}/*.png
    /home/teaching/group46/CT_dataset/{train,val}/{patient_id}/*.png

  Inference (test):
    test/{patient}/CT/*.png  +  test/{patient}/MR/*.png

Output (ALL results saved under a single --out_dir):
    out_dir/
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── latest.pth
    ├── inference/
    │   ├── fused/{patient_id}/*.png
    │   ├── grids/{patient_id}/*.png
    │   ├── metrics.csv
    │   └── summary.txt
    └── val_metrics.csv

Usage:
    # Train only
    python fusion_pipeline.py --mode train

    # Inference only (requires --ckpt)
    python fusion_pipeline.py --mode infer \
        --ckpt /path/to/best_model.pth \
        --test_dir /path/to/test

    # Train then immediately run inference on test set
    python fusion_pipeline.py --mode both \
        --test_dir /path/to/test
"""

import os
import csv
import math
import argparse
from pathlib import Path
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ssim, ms_ssim
from timm.models.vision_transformer import PatchEmbed, Block
from piq import vif_p


# ─────────────────────────────────────────────────────────────────────────────
# 1.  APFM  — Adaptive Pooling Fusion Module
# ─────────────────────────────────────────────────────────────────────────────
class APFM(nn.Module):
    """Fuses two feature maps of identical shape via learned channel+spatial gates."""
    def __init__(self, in_channels: int):
        super().__init__()
        g = min(32, in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1   = nn.Conv2d(in_channels * 4, 2 * in_channels, 1)
        self.norm1   = nn.GroupNorm(g, 2 * in_channels)
        self.silu    = nn.SiLU()
        self.conv2   = nn.Conv2d(2 * in_channels, in_channels, 1)
        self.norm2   = nn.GroupNorm(g, in_channels)
        self.conv3   = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.norm3   = nn.GroupNorm(g, in_channels)
        self.silu2   = nn.SiLU()
        self.conv4   = nn.Conv2d(in_channels, in_channels, 1)
        self.norm4   = nn.GroupNorm(g, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.last_gate_mean = None   # monitored during training

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy     = torch.cat([x, y], dim=1)
        avg    = self.global_avg_pool(xy)
        mx     = self.global_max_pool(xy)
        pooled = torch.cat([avg, mx], dim=1)
        w_ch   = self.norm2(self.conv2(self.silu(self.norm1(self.conv1(pooled)))))
        w_sp   = self.norm4(self.conv4(self.silu2(self.norm3(self.conv3(xy)))))
        gate   = self.sigmoid(w_ch + w_sp)
        self.last_gate_mean = gate.mean().detach()
        return x * gate + y * (1.0 - gate)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAE ViT-Small Encoder  (frozen after loading)
# ─────────────────────────────────────────────────────────────────────────────
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    assert embed_dim % 4 == 0
    omega  = 1.0 / (10000 ** (np.arange(embed_dim // 4, dtype=np.float32) / (embed_dim / 4)))

    def embed_1d(g, o):
        g   = g.reshape(-1)
        out = np.einsum('m,d->md', g, o)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([embed_1d(grid[0], omega), embed_1d(grid[1], omega)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


class MAEEncoder(nn.Module):
    """MAE ViT-Small encoder; returns per-block intermediate features."""
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
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        pe = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x   = self.patch_embed(x)
        x   = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, 1:, :])
        x = self.norm(x)
        feats[-1] = x[:, 1:, :]
        return feats   # 12 × (B, 256, 384)


def load_frozen_encoder(ckpt_path: str, device: torch.device,
                        tag: str = '') -> MAEEncoder:
    enc   = MAEEncoder()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    enc_keys = {k: v for k, v in state.items()
                if k.startswith(('patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm'))}
    missing, unexpected = enc.load_state_dict(enc_keys, strict=False)
    print(f"  [{tag} encoder] missing={len(missing)}, unexpected={len(unexpected)}")
    enc.to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAE ViT-Small Decoder  (trainable)
# ─────────────────────────────────────────────────────────────────────────────
class MAEDecoder(nn.Module):
    """ViT decoder with symmetric skip connections from fused encoder layers."""
    def __init__(self, num_patches=256, embed_dim=384,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 patch_size=16, in_chans=1,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.decoder_embed     = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)
        self.skip_proj    = nn.Linear(embed_dim, decoder_embed_dim)
        self._init_weights(num_patches)

    def _init_weights(self, num_patches: int):
        pe = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, fused_tokens: torch.Tensor,
                skip_feats: List[torch.Tensor]) -> torch.Tensor:
        B, L, _ = fused_tokens.shape
        x       = self.decoder_embed(fused_tokens)
        cls_dec = torch.zeros(B, 1, x.shape[-1], device=x.device)
        x       = torch.cat([cls_dec, x], dim=1) + self.decoder_pos_embed
        for i, blk in enumerate(self.decoder_blocks):
            if i < len(skip_feats):
                skip = self.skip_proj(skip_feats[i])
                skip = torch.cat([torch.zeros(B, 1, skip.shape[-1], device=x.device),
                                   skip], dim=1)
                x = x + skip
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])
        p = 16; h = w = int(L ** .5)
        x = x.reshape(B, h, w, p, p, 1)
        x = torch.einsum('bhwpqc->bchpwq', x).reshape(B, 1, h * p, w * p)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Fusion Model
# ─────────────────────────────────────────────────────────────────────────────
class CTMRIFusionModel(nn.Module):
    """
    Frozen CT-encoder + frozen MRI-encoder →
    per-layer APFM fusion (12 layers) →
    trainable MAE-ViT decoder with symmetric skips.
    """
    ENCODER_DEPTH = 12
    DECODER_DEPTH = 8

    def __init__(self, ct_ckpt: str = None, mri_ckpt: str = None,
                 device: torch.device = None):
        super().__init__()
        embed_dim        = 384
        self.ct_enc      = MAEEncoder()
        self.mri_enc     = MAEEncoder()
        self.apfm_layers = nn.ModuleList([APFM(embed_dim) for _ in range(self.ENCODER_DEPTH)])
        self.decoder     = MAEDecoder(
            num_patches=256, embed_dim=embed_dim,
            decoder_embed_dim=512, decoder_depth=self.DECODER_DEPTH,
            decoder_num_heads=16, patch_size=16, in_chans=1)

        if ct_ckpt and mri_ckpt and device is not None:
            self._load_encoders(ct_ckpt, mri_ckpt, device)

    def _load_encoders(self, ct_ckpt: str, mri_ckpt: str, device: torch.device):
        print("Loading frozen encoders...")
        enc_ct  = load_frozen_encoder(ct_ckpt,  device, 'CT')
        enc_mri = load_frozen_encoder(mri_ckpt, device, 'MRI')
        self.ct_enc.load_state_dict(enc_ct.state_dict())
        self.mri_enc.load_state_dict(enc_mri.state_dict())
        for p in self.ct_enc.parameters():
            p.requires_grad_(False)
        for p in self.mri_enc.parameters():
            p.requires_grad_(False)
        self.ct_enc.eval()
        self.mri_enc.eval()

    def _tokens_to_spatial(self, t: torch.Tensor) -> torch.Tensor:
        B, L, C = t.shape; g = int(L ** .5)
        return t.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()

    def _spatial_to_tokens(self, f: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f.shape
        return f.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    def forward(self, ct: torch.Tensor, mri: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ct_feats  = self.ct_enc(ct)
            mri_feats = self.mri_enc(mri)
        fused_layers = []
        for i, apfm in enumerate(self.apfm_layers):
            f = apfm(self._tokens_to_spatial(ct_feats[i]),
                     self._tokens_to_spatial(mri_feats[i]))
            fused_layers.append(self._spatial_to_tokens(f))
        final_fused = fused_layers[self.ENCODER_DEPTH - 1]
        skip_feats  = [fused_layers[self.ENCODER_DEPTH - 2 - i]
                       for i in range(self.DECODER_DEPTH)]
        return self.decoder(final_fused, skip_feats)


def build_model_for_inference(ckpt_path: str, ct_ckpt: str, mri_ckpt: str,
                               device: torch.device) -> CTMRIFusionModel:
    """Three-part restore: encoder weights + APFM/decoder weights."""
    print("\nBuilding model for inference...")
    model = CTMRIFusionModel()
    model._load_encoders(ct_ckpt, mri_ckpt, device)

    print(f"  Loading fusion checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        full_state = ckpt['model']
        print(f"    Epoch: {ckpt.get('epoch','?')}  "
              f"Best PSNR: {ckpt.get('best_psnr','?'):.4f}")
    else:
        full_state = ckpt

    missing, unexpected = model.load_state_dict(full_state, strict=False)
    non_enc_missing = [k for k in missing
                       if not k.startswith(('ct_enc.', 'mri_enc.'))]
    if non_enc_missing:
        print(f"    [WARN] Non-encoder missing keys: {len(non_enc_missing)}")
    model.to(device).eval()
    print(f"  Model ready on {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Datasets
# ─────────────────────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """Training/validation dataset: mirrored CT and MRI folder trees."""
    def __init__(self, ct_root: str, mri_root: str, split: str = 'train'):
        self.ct_root  = Path(ct_root)  / split
        self.mri_root = Path(mri_root) / split
        self.pairs: List[Tuple[Path, Path, str]] = []

        ct_patients  = sorted(p.name for p in self.ct_root.iterdir()  if p.is_dir())
        mri_patients = sorted(p.name for p in self.mri_root.iterdir() if p.is_dir())
        shared       = sorted(set(ct_patients) & set(mri_patients))

        for pid in shared:
            ct_slices  = sorted((self.ct_root  / pid).glob('*.png'))
            mri_slices = sorted((self.mri_root / pid).glob('*.png'))
            n = min(len(ct_slices), len(mri_slices))
            for i in range(n):
                self.pairs.append((ct_slices[i], mri_slices[i], pid))

        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        print(f"[Dataset/{split}] {len(self.pairs)} pairs from {len(shared)} patients")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ct_p, mri_p, pid = self.pairs[idx]
        return self.tf(Image.open(ct_p)), self.tf(Image.open(mri_p)), pid


class TestDataset(Dataset):
    """Inference dataset: test/{patient}/CT/*.png + MR/*.png"""
    CT_SUBFOLDER_CANDIDATES  = ['CT',  'ct',  'Ct']
    MRI_SUBFOLDER_CANDIDATES = ['MR',  'MRI', 'mr', 'mri', 'Mr']

    def __init__(self, test_dir: str = None,
                 ct_dir: str = None, mri_dir: str = None):
        self.pairs: List[Tuple[Path, Path, str, str]] = []
        if test_dir is not None:
            self._load_from_test_dir(Path(test_dir))
        elif ct_dir is not None and mri_dir is not None:
            self._load_from_separate_dirs(Path(ct_dir), Path(mri_dir))
        else:
            raise ValueError("Provide --test_dir or both --ct_dir and --mri_dir")
        if not self.pairs:
            raise RuntimeError("No paired PNG images found.")
        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def _load_from_test_dir(self, root: Path):
        if not root.exists():
            raise FileNotFoundError(f"test_dir not found: {root}")
        loaded = 0
        for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            pid     = patient_dir.name
            ct_sub  = self._find_subdir(patient_dir, self.CT_SUBFOLDER_CANDIDATES)
            mri_sub = self._find_subdir(patient_dir, self.MRI_SUBFOLDER_CANDIDATES)
            if ct_sub is None or mri_sub is None:
                print(f"  [SKIP] {pid}: missing CT or MR sub-folder"); continue
            cts  = sorted(ct_sub.glob('*.png'))
            mris = sorted(mri_sub.glob('*.png'))
            if not cts or not mris:
                print(f"  [SKIP] {pid}: empty folder"); continue
            n = min(len(cts), len(mris))
            if len(cts) != len(mris):
                print(f"  [WARN] {pid}: CT={len(cts)}, MR={len(mris)}, using {n}")
            for c, m in zip(cts[:n], mris[:n]):
                self.pairs.append((c, m, pid, c.name))
            loaded += 1
        print(f"[TestDataset] {loaded} patients, {len(self.pairs)} pairs")

    def _load_from_separate_dirs(self, ct_root: Path, mri_root: Path):
        for root in (ct_root, mri_root):
            if not root.exists():
                raise FileNotFoundError(f"Not found: {root}")
        ct_subs  = sorted(p for p in ct_root.iterdir()  if p.is_dir())
        mri_subs = sorted(p for p in mri_root.iterdir() if p.is_dir())
        if ct_subs and mri_subs:
            shared = sorted(set(p.name for p in ct_subs) & set(p.name for p in mri_subs))
            if shared:
                for pid in shared:
                    for c, m in zip(sorted((ct_root  / pid).glob('*.png')),
                                    sorted((mri_root / pid).glob('*.png'))):
                        self.pairs.append((c, m, pid, c.name))
                print(f"[TestDataset] {len(shared)} patients, {len(self.pairs)} pairs")
                return
        for c, m in zip(sorted(ct_root.glob('*.png')), sorted(mri_root.glob('*.png'))):
            self.pairs.append((c, m, 'test', c.name))
        print(f"[TestDataset] Flat: {len(self.pairs)} pairs")

    @staticmethod
    def _find_subdir(parent: Path, candidates: List[str]):
        for name in candidates:
            p = parent / name
            if p.is_dir(): return p
        return None

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ct_p, mri_p, pid, name = self.pairs[idx]
        return self.tf(Image.open(ct_p)), self.tf(Image.open(mri_p)), pid, name


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Loss  — FIXED: balanced CT/MRI contribution
# ─────────────────────────────────────────────────────────────────────────────
class GradientLoss(nn.Module):
    """Sobel-based gradient loss with explicit CT edge preservation."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, ref_ct, ref_mri):
        def grad_mag(x):
            gx = F.conv2d(x, self.kx, padding=1)
            gy = F.conv2d(x, self.ky, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)
        gp   = grad_mag(pred)
        gct  = grad_mag(ref_ct)
        gmri = grad_mag(ref_mri)
        g_ref = torch.max(gct, gmri)
        l_combined = F.l1_loss(gp, g_ref)
        # Preserve CT-specific edges (where CT has edges but MRI does not)
        ct_only = F.relu(gct - gmri)
        mask    = (ct_only > 0.01).float()
        l_ct_edge = F.l1_loss(gp * mask, gct * mask) if mask.sum() > 0 else gp.new_tensor(0.)
        return l_combined + 0.5 * l_ct_edge


class DynamicModalityLoss(nn.Module):
    """
    FIX: Balanced independent losses instead of variance-weighted averaging.
    Previous version gave near-zero CT weight wherever MRI variance > CT variance,
    causing the model to output essentially MRI (6.6 dB PSNR gap).
    """
    def forward(self, pred, ct, mri):
        loss_ct  = F.l1_loss(pred, ct)
        loss_mri = F.l1_loss(pred, mri)
        # Rescale each term by the other's magnitude so neither can dominate
        scale_ct  = (loss_mri.detach() / (loss_ct.detach()  + 1e-8)).clamp(0.5, 2.0)
        scale_mri = (loss_ct.detach()  / (loss_mri.detach() + 1e-8)).clamp(0.5, 2.0)
        return scale_ct * loss_ct + scale_mri * loss_mri


class FusionLoss(nn.Module):
    """
    Balanced Fusion Loss:
      L = λ_int·L_int + λ_ssim·L_ssim + λ_msssim·L_msssim
          + λ_grad·L_grad + λ_modal·L_modal
    """
    def __init__(self, λ_int=1.0, λ_ssim=0.5, λ_msssim=0.5,
                 λ_grad=0.5, λ_modal=0.3):
        super().__init__()
        self.λ_int    = λ_int
        self.λ_ssim   = λ_ssim
        self.λ_msssim = λ_msssim
        self.λ_grad   = λ_grad
        self.λ_modal  = λ_modal
        self.grad_loss  = GradientLoss()
        self.modal_loss = DynamicModalityLoss()

    def forward(self, pred, ct, mri):
        """All tensors: (B, 1, 256, 256) in [0,1]."""
        # L_int: equal MAE vs both modalities
        l_int = 0.5 * (F.l1_loss(pred, ct) + F.l1_loss(pred, mri))

        # L_ssim: equal weight, plus extra CT penalty to counter historical bias
        l_ssim = (
            (1.0 - ssim(pred, ct,  data_range=1.0, size_average=True)) +
            (1.0 - ssim(pred, mri, data_range=1.0, size_average=True))
        ) * 0.5
        # Extra CT SSIM penalty
        l_ssim = l_ssim + 0.3 * (1.0 - ssim(pred, ct, data_range=1.0, size_average=True))

        # L_msssim
        l_msssim = 1.0 - 0.5 * (
            ms_ssim(pred, ct,  data_range=1.0, size_average=True) +
            ms_ssim(pred, mri, data_range=1.0, size_average=True))

        # L_grad (with CT-edge preservation)
        l_grad = self.grad_loss(pred, ct, mri)

        # L_modal (balanced — core fix)
        l_modal = self.modal_loss(pred, ct, mri)

        total = (self.λ_int    * l_int   +
                 self.λ_ssim   * l_ssim  +
                 self.λ_msssim * l_msssim +
                 self.λ_grad   * l_grad  +
                 self.λ_modal  * l_modal)

        return total, dict(int=l_int.item(), ssim=l_ssim.item(),
                           msssim=l_msssim.item(), grad=l_grad.item(),
                           modal=l_modal.item())


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()


def _entropy(arr: np.ndarray) -> float:
    hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=False)
    hist     = hist.astype(np.float64) + 1e-10
    hist    /= hist.sum()
    return float(-(hist * np.log2(hist)).sum())


def compute_metrics(pred: torch.Tensor,
                    ct:   torch.Tensor,
                    mri:  torch.Tensor) -> Dict[str, float]:
    p   = pred.clamp(0, 1)
    c   = ct.clamp(0, 1)
    m   = mri.clamp(0, 1)
    r   = 0.5 * (c + m)
    arr = p.squeeze(1).cpu().numpy()
    return dict(
        psnr_ct     = _psnr(p, c),
        ssim_ct     = ssim(p, c, data_range=1., size_average=True).item(),
        msssim_ct   = ms_ssim(p, c, data_range=1., size_average=True).item(),
        psnr_mri    = _psnr(p, m),
        ssim_mri    = ssim(p, m, data_range=1., size_average=True).item(),
        msssim_mri  = ms_ssim(p, m, data_range=1., size_average=True).item(),
        psnr_mean   = _psnr(p, r),
        ssim_mean   = ssim(p, r, data_range=1., size_average=True).item(),
        msssim_mean = ms_ssim(p, r, data_range=1., size_average=True).item(),
        vif         = vif_p(p, r, data_range=1.).item(),
        mse         = F.mse_loss(p, r).item(),
        mae         = F.l1_loss(p, r).item(),
        sd          = float(arr.std()),
        en          = _entropy(arr),
        overall     = (_psnr(p, r) / 50.0 +
                       ssim(p, r, data_range=1., size_average=True).item() +
                       ms_ssim(p, r, data_range=1., size_average=True).item() +
                       vif_p(p, r, data_range=1.).item()) / 4.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Training Utilities
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for step, (ct, mri, _) in enumerate(loader):
        ct, mri = ct.to(device), mri.to(device)
        pred    = model(ct, mri)
        loss, comps = criterion(pred, ct, mri)

        # Gate balance penalty — keeps APFM from locking to one modality
        gate_means = [apfm.last_gate_mean for apfm in model.apfm_layers
                      if apfm.last_gate_mean is not None]
        if gate_means:
            gate_balance = sum((g - 0.5).pow(2) for g in gate_means) / len(gate_means)
            loss = loss + 0.1 * gate_balance

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % 50 == 0:
            comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comps.items())
            # Also report gate balance
            if gate_means:
                avg_gate = sum(g.item() for g in gate_means) / len(gate_means)
                comp_str += f"  gate={avg_gate:.3f}"
            print(f"  [E{epoch} {step}/{len(loader)}] loss={loss.item():.4f}  {comp_str}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, csv_path: str, best_psnr: float):
    model.eval()
    rows     = []
    val_loss = 0.0
    agg      = {k: 0.0 for k in ('psnr_ct','ssim_ct','msssim_ct',
                                   'psnr_mri','ssim_mri','msssim_mri',
                                   'psnr_mean','ssim_mean','msssim_mean',
                                   'vif','sd','en','mse','mae','overall')}

    for ct, mri, pids in loader:
        ct, mri = ct.to(device), mri.to(device)
        pred    = model(ct, mri)
        loss, _ = criterion(pred, ct, mri)
        val_loss += loss.item()
        m = compute_metrics(pred, ct, mri)
        for k in agg:
            agg[k] += m[k]
        for i, pid in enumerate(pids):
            pm = compute_metrics(pred[i:i+1], ct[i:i+1], mri[i:i+1])
            rows.append({'patient': pid, **pm})

    n = len(loader)
    for k in agg: agg[k] /= n

    # Gate stats
    gate_vals = [apfm.last_gate_mean.item() for apfm in model.apfm_layers
                 if apfm.last_gate_mean is not None]
    if gate_vals:
        print(f"  APFM gate  min={min(gate_vals):.3f}  "
              f"max={max(gate_vals):.3f}  avg={sum(gate_vals)/len(gate_vals):.3f}  "
              f"(target ~0.5)")

    fieldnames = ['patient','psnr_ct','ssim_ct','msssim_ct',
                  'psnr_mri','ssim_mri','msssim_mri',
                  'psnr_mean','ssim_mean','msssim_mean',
                  'vif','mse','mae','sd','en','overall']
    write_header = not Path(csv_path).exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.6f}" if k != 'patient' else row[k]
                             for k in fieldnames})

    print(f"  [Val E{epoch}] loss={val_loss/n:.4f}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    return val_loss / n, agg['psnr_mean']


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Inference Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def save_grid(ct, mri, pred, metrics, save_path: Path, title: str = ''):
    IMG_W, IMG_H = 256, 256
    PAD, TITLE_H, METRIC_H, COLS = 12, 32, 210, 3
    CW = COLS * IMG_W + (COLS + 1) * PAD
    CH = TITLE_H + IMG_H + METRIC_H + 2 * PAD
    canvas = Image.new('RGB', (CW, CH), color=(240, 242, 248))
    draw   = ImageDraw.Draw(canvas)

    FONTS_BOLD = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    FONTS_REG  = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    ft = fm = None
    for fb, fr in zip(FONTS_BOLD, FONTS_REG):
        try:
            ft = ImageFont.truetype(fb, 14)
            fm = ImageFont.truetype(fr, 10)
            break
        except Exception:
            continue
    if ft is None:
        ft = fm = ImageFont.load_default()

    if title:
        draw.text((PAD, 6), title, fill=(30, 30, 30), font=ft)

    col_labels = ['CT  (input)', 'MRI  (input)', 'Fused  (output)']
    col_images = [_to_pil(ct), _to_pil(mri), _to_pil(pred.clamp(0, 1))]
    label_clrs = [(50, 50, 50), (50, 50, 50), (10, 50, 150)]

    for ci, (label, img, lc) in enumerate(zip(col_labels, col_images, label_clrs)):
        x0    = PAD + ci * (IMG_W + PAD)
        y_img = TITLE_H + PAD
        draw.text((x0 + 4, TITLE_H - 16), label, fill=lc, font=ft)
        canvas.paste(img.convert('RGB'), (x0, y_img))
        y_sep = y_img + IMG_H + 4
        draw.line([(x0, y_sep), (x0 + IMG_W, y_sep)], fill=(170, 170, 180), width=1)
        if ci == 2:
            y = y_sep + 8
            for line in [
                "── vs CT ──────────────────────────",
                f"  PSNR    : {metrics['psnr_ct']:.2f} dB",
                f"  SSIM    : {metrics['ssim_ct']:.4f}",
                f"  MS-SSIM : {metrics['msssim_ct']:.4f}",
                "── vs MRI ─────────────────────────",
                f"  PSNR    : {metrics['psnr_mri']:.2f} dB",
                f"  SSIM    : {metrics['ssim_mri']:.4f}",
                f"  MS-SSIM : {metrics['msssim_mri']:.4f}",
                "── vs Mean (CT+MRI)/2 ─────────────",
                f"  PSNR    : {metrics['psnr_mean']:.2f} dB",
                f"  SSIM    : {metrics['ssim_mean']:.4f}",
                f"  VIF     : {metrics['vif']:.4f}",
                "── Image Quality ───────────────────",
                f"  SD      : {metrics['sd']:.4f}",
                f"  EN      : {metrics['en']:.4f}",
                f"  OVERALL : {metrics['overall']:.4f}",
            ]:
                clr = (0, 70, 160) if line.startswith('──') else (5, 5, 110)
                draw.text((x0 + 5, y), line, fill=clr, font=fm)
                y += 12
        else:
            draw.text((x0 + 5, y_sep + 90), "metrics →", fill=(160, 160, 160), font=fm)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(save_path))


@torch.no_grad()
def run_inference(model, dataset, device, infer_dir: Path,
                  batch_size: int = 4) -> Dict[str, float]:
    """Run inference; all outputs saved under infer_dir/."""
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    fused_dir = infer_dir / 'fused'
    grids_dir = infer_dir / 'grids'
    fused_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    csv_path   = infer_dir / 'metrics.csv'
    fieldnames = ['patient', 'image',
                  'psnr_ct',   'ssim_ct',   'msssim_ct',
                  'psnr_mri',  'ssim_mri',  'msssim_mri',
                  'psnr_mean', 'ssim_mean', 'msssim_mean',
                  'vif', 'mse', 'mae', 'sd', 'en', 'overall']

    agg: Dict[str, float] = {}
    total = 0

    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, (ct, mri, pids, names) in enumerate(loader):
            ct, mri = ct.to(device), mri.to(device)
            pred    = model(ct, mri).clamp(0, 1)

            for i in range(ct.shape[0]):
                pid, name             = pids[i], names[i]
                stem                  = Path(name).stem
                ct_i, mri_i, pred_i   = ct[i], mri[i], pred[i]

                m = compute_metrics(pred_i.unsqueeze(0),
                                    ct_i.unsqueeze(0),
                                    mri_i.unsqueeze(0))
                if not agg:
                    agg = {k: 0.0 for k in m}
                for k in agg: agg[k] += m[k]
                total += 1

                # Fused PNG
                fused_np   = (pred_i.squeeze(0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                fused_save = fused_dir / pid / f"{stem}.png"
                fused_save.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(fused_np, mode='L').save(str(fused_save))

                # Visual grid
                save_grid(ct_i, mri_i, pred_i, m,
                          grids_dir / pid / f"{stem}.png",
                          title=f"{pid} / {name}")

                writer.writerow({'patient': pid, 'image': name,
                                 **{k: f"{v:.6f}" for k, v in m.items()}})

            done = min((batch_idx + 1) * loader.batch_size, len(dataset))
            print(f"  Processed {done}/{len(dataset)} ...", end='\r')

        # OVERALL summary row
        avg = {k: v / total for k, v in agg.items()}
        writer.writerow({'patient': 'OVERALL', 'image': f'({total} images)',
                         **{k: f"{v:.6f}" for k, v in avg.items()}})

    print(f"\n  Done. {total} images processed.")
    for k in agg: agg[k] /= total
    return agg


def save_summary(agg: Dict[str, float], infer_dir: Path,
                 ckpt_path: str, ct_ckpt: str, mri_ckpt: str, data_info: str):
    lines = [
        "=" * 58,
        "  CT-MRI FUSION — INFERENCE SUMMARY",
        "=" * 58,
        f"  Fusion ckpt : {ckpt_path}",
        f"  CT  encoder : {ct_ckpt}",
        f"  MRI encoder : {mri_ckpt}",
        f"  Data        : {data_info}",
        "",
        "  ── vs CT ────────────────────────────────",
        f"  PSNR    : {agg['psnr_ct']:.4f} dB",
        f"  SSIM    : {agg['ssim_ct']:.4f}",
        f"  MS-SSIM : {agg['msssim_ct']:.4f}",
        "",
        "  ── vs MRI ───────────────────────────────",
        f"  PSNR    : {agg['psnr_mri']:.4f} dB",
        f"  SSIM    : {agg['ssim_mri']:.4f}",
        f"  MS-SSIM : {agg['msssim_mri']:.4f}",
        "",
        f"  PSNR gap (MRI-CT): {agg['psnr_mri'] - agg['psnr_ct']:.2f} dB  "
        f"(target <3 dB)",
        "",
        "  ── vs Mean (CT+MRI)/2 ───────────────────",
        f"  PSNR    : {agg['psnr_mean']:.4f} dB",
        f"  SSIM    : {agg['ssim_mean']:.4f}",
        f"  MS-SSIM : {agg['msssim_mean']:.4f}",
        f"  VIF     : {agg['vif']:.4f}",
        f"  MSE     : {agg['mse']:.6f}",
        f"  MAE     : {agg['mae']:.6f}",
        "",
        "  ── Image Quality ────────────────────────",
        f"  SD      : {agg['sd']:.4f}",
        f"  EN      : {agg['en']:.4f}",
        f"  OVERALL : {agg['overall']:.4f}",
        "=" * 58,
    ]
    txt = '\n'.join(lines)
    print('\n' + txt)
    sp = infer_dir / 'summary.txt'
    sp.write_text(txt + '\n')
    print(f"\n  Summary      → {sp}")
    print(f"  Metrics CSV  → {infer_dir}/metrics.csv")
    print(f"  Fused images → {infer_dir}/fused/")
    print(f"  Visual grids → {infer_dir}/grids/")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Training Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def do_train(args, device: torch.device):
    out_dir   = Path(args.out_dir)
    ckpt_dir  = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = str(out_dir / 'val_metrics.csv')

    train_ds = FusionDataset(args.ct_root, args.mri_root, 'train')
    val_ds   = FusionDataset(args.ct_root, args.mri_root, 'val')
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model     = CTMRIFusionModel(args.ct_ckpt, args.mri_ckpt, device).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FusionLoss().to(device)

    start_epoch = 1
    best_psnr   = -1.0

    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -1.0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    early_stopper = EarlyStopping(patience=10)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}   lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, epoch)
        val_loss, avg_psnr = validate(model, val_dl, criterion, device, epoch,
                                      csv_path, best_psnr)
        scheduler.step()

        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_psnr=best_psnr,
                    train_loss=train_loss, val_loss=val_loss)
        torch.save(ckpt, ckpt_dir / 'latest.pth')

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(ckpt, ckpt_dir / 'best_model.pth')
            print(f"  ★ Best model saved (PSNR={best_psnr:.4f})")

        if early_stopper.step(avg_psnr):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f}")
    return str(ckpt_dir / 'best_model.pth')


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Inference Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def do_infer(args, device: torch.device, ckpt_path: str = None):
    out_dir   = Path(args.out_dir)
    infer_dir = out_dir / 'inference'
    infer_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_path or args.ckpt
    if not ckpt_path:
        raise ValueError("Inference requires --ckpt or prior training run")

    model = build_model_for_inference(ckpt_path, args.ct_ckpt, args.mri_ckpt, device)

    print(f"\nScanning test data...")
    if args.test_dir:
        dataset   = TestDataset(test_dir=args.test_dir)
        data_info = args.test_dir
    else:
        dataset   = TestDataset(ct_dir=args.ct_dir, mri_dir=args.mri_dir)
        data_info = f"CT={args.ct_dir}  MRI={args.mri_dir}"
    print(f"  Total pairs : {len(dataset)}")

    print(f"\nRunning inference → {infer_dir}")
    agg = run_inference(model, dataset, device, infer_dir, batch_size=args.batch)
    save_summary(agg, infer_dir, ckpt_path, args.ct_ckpt, args.mri_ckpt, data_info)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Argument Parser + Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='CT-MRI Fusion Pipeline')

    p.add_argument('--mode', choices=['train', 'infer', 'both'], default='train',
                   help='train | infer | both (train then infer)')

    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument('--ct_ckpt',
                   default='/home/teaching/group46/MAEDiff-main/mae_vit_small_patch16_ct/encoder.pth')
    p.add_argument('--mri_ckpt',
                   default='/home/teaching/group46/MAEDiff-main/mae_vit_small_patch16_mri/encoder.pth')
    p.add_argument('--ckpt', default='',
                   help='Fusion checkpoint for inference-only mode')
    p.add_argument('--resume', default='',
                   help='Resume training from this checkpoint')

    # ── Data — Training ───────────────────────────────────────────────────────
    p.add_argument('--ct_root',  default='/home/teaching/group46/CT_dataset')
    p.add_argument('--mri_root', default='/home/teaching/group46/MRI_dataset')

    # ── Data — Inference ──────────────────────────────────────────────────────
    p.add_argument('--test_dir', default='/home/teaching/group46/attempt_4/test',
                   help='test/{patient}/CT/*.png + MR/*.png')
    p.add_argument('--ct_dir',  default=None, help='(legacy) flat CT dir')
    p.add_argument('--mri_dir', default=None, help='(legacy) flat MRI dir')

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', default='/home/teaching/group46/MAEDiff-main/fusion_output_dynloss',
                   help='Single root for ALL outputs: checkpoints/, inference/, val_metrics.csv')

    # ── Training Hyper-params ─────────────────────────────────────────────────
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num_workers', type=int,   default=4)

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Mode    : {args.mode}")
    print(f"Out dir : {args.out_dir}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        do_train(args, device)

    elif args.mode == 'infer':
        do_infer(args, device)

    elif args.mode == 'both':
        best_ckpt = do_train(args, device)
        do_infer(args, device, ckpt_path=best_ckpt)


if __name__ == '__main__':
    main()
    
# python fusion_pipeline_v2.py --mode infer --ckpt /home/teaching/group46/MAEDiff-main/fusion_output/checkpoints/best_model.pth