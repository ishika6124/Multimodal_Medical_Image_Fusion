"""
extract_features.py — MedSAM2D CT Encoder Feature Extractor
=============================================================
Paths (hardcoded for this project):
    Checkpoint : /home/teaching/group46/SAM-Med2D/pretrain_model/sam-med2d_b.pth
    CT dataset : /home/teaching/group46/CT_dataset/{train|val}/patient_id/*.png
    Output     : /home/teaching/group46/CT_embeddings/{train|val}/

What this script does:
    1. Loads the SAM-Med2D pretrained ViT-B image encoder (frozen)
    2. Passes every CT slice (PNG) through the encoder
    3. Saves each embedding as a .npy file  → shape (256, 64, 64)
    4. Trains a tiny decoder (5 epochs) to reconstruct images from embeddings
    5. Computes PSNR, SSIM, MS-SSIM, VIF, SD, EN, MSE, MAE
    6. Saves metrics_summary.json and metrics.csv

Run:
    cd /home/teaching/group46/SAM-Med2D
    python extract_features.py --split train
    python extract_features.py --split val
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ── scikit-image metrics ──────────────────────────────────────────────────── #
from skimage.metrics import (
    peak_signal_noise_ratio as sk_psnr,
    structural_similarity   as sk_ssim,
    mean_squared_error      as sk_mse,
)

# ── sewar for MS-SSIM and VIF ─────────────────────────────────────────────── #
try:
    from sewar.full_ref import vifp, msssim
    HAS_SEWAR = True
except ImportError:
    print("[!] sewar not installed → MS-SSIM and VIF will be NaN")
    print("    Fix: pip install sewar")
    HAS_SEWAR = False


# ============================================================================ #
#   PATHS  — change only these if your layout changes
# ============================================================================ #
REPO_ROOT   = Path("/home/teaching/group46/SAM-Med2D")
CHECKPOINT  = REPO_ROOT / "pretrain_model" / "sam-med2d_b.pth"
MRI_DATASET  = Path("/home/teaching/group46/MRI_dataset")
OUTPUT_ROOT = Path("/home/teaching/group46/MRI_embeddings")


# ============================================================================ #
#   1.  LOAD ENCODER
# ============================================================================ #

def load_encoder(checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT))

    from segment_anything import sam_model_registry

    print(f"[→] Loading checkpoint: {checkpoint}")
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

    # Build a simple args namespace matching what build_sam_vit_b expects
    import argparse
    args = argparse.Namespace(
        image_size      = 256,
        sam_checkpoint  = str(checkpoint),
        encoder_adapter = True,   # SAM-Med2D uses adapter layers
    )

    sam = sam_model_registry["vit_b"](args)
    sam.to(device).eval()

    encoder = sam.image_encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Shape check
    dummy = torch.zeros(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"[✓] Encoder loaded. Output shape: {tuple(out.shape)}")
    return encoder

# ============================================================================ #
#   2.  DATASET
# ============================================================================ #

class MRIDataset(torch.utils.data.Dataset):
    """
    Walks CT_dataset/{split}/patient_id/*.png and returns every slice.

    Structure:
        CT_dataset/
            train/
                1BA001/
                    000.png
                    001.png
            val/
                ...
    """

    # SAM-Med2D was trained with ImageNet normalisation on 3-channel input
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: str, image_size: int = 256):
        self.split      = split
        self.image_size = image_size
        split_dir       = MRI_DATASET / split

        assert split_dir.exists(), f"Split folder not found: {split_dir}"

        # Collect all PNG paths
        self.paths = sorted(
            p for p in split_dir.rglob("*.png")
        )
        if not self.paths:
            # also try jpg/jpeg
            self.paths = sorted(
                p for p in split_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg"}
            )

        assert self.paths, f"No images found in {split_dir}"
        print(f"[✓] {split} split: {len(self.paths)} CT slices found")

        # Transform for encoder input (RGB, normalised)
        self.enc_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # CT is grey → repeat to 3ch
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        # Raw single-channel for metric ground truth
        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),           # [0,1] float32
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(str(path)).convert("RGB")
        return {
            "enc": self.enc_transform(img),       # (3, H, W)  for encoder
            "raw": self.raw_transform(img),        # (1, H, W)  for metrics
            "path": str(path),
            "patient": path.parent.name,           # e.g. 1BA001
        }


# ============================================================================ #
#   3.  TINY DECODER  (for reconstruction → metric computation)
# ============================================================================ #

class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16→32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2),  # 32→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2),  # 64→128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,  16, kernel_size=2, stride=2),  # 128→256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        return self.net(feat)


def train_decoder(encoder, decoder, dataset, device,
                  epochs=5, batch_size=16, lr=1e-3):
    """
    Quickly trains the decoder to reconstruct CT slices from encoder embeddings.
    Encoder stays frozen throughout.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)
    decoder.train()

    print(f"\n[→] Training tiny decoder ({epochs} epochs) ...")
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in tqdm(loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            imgs = batch["enc"].to(device)   # (B,3,256,256)
            gts  = batch["raw"].to(device)   # (B,1,256,256)

            with torch.no_grad():
                feats = encoder(imgs)        # (B,256,64,64)

            preds = decoder(feats)           # (B,1,256,256)
            loss  = F.mse_loss(preds, gts)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item()

        print(f"    Epoch {epoch}/{epochs}  loss={running/len(loader):.5f}")

    decoder.eval()
    print("[✓] Decoder ready\n")
    return decoder


# ============================================================================ #
#   4.  METRICS
# ============================================================================ #

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    pred, target : float32 arrays in [0,1], shape (H, W)
    Returns      : dict with PSNR, SSIM, MS-SSIM, VIF, SD, EN, MSE, MAE
    """
    pred_u8   = (np.clip(pred,   0, 1) * 255).astype(np.uint8)
    target_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)

    mse_val  = float(sk_mse(target, pred))
    mae_val  = float(np.mean(np.abs(pred - target)))
    psnr_val = float(sk_psnr(target, pred, data_range=1.0))

    # SSIM needs win_size ≤ min(H,W); default 7 is fine for 256×256
    ssim_val = float(sk_ssim(target, pred, data_range=1.0))

    # MS-SSIM (sewar expects uint8)
    if HAS_SEWAR:
        try:
            msssim_val = float(msssim(target_u8, pred_u8).real)
        except Exception:
            msssim_val = float("nan")
        try:
            vif_val = float(vifp(target_u8, pred_u8))
        except Exception:
            vif_val = float("nan")
    else:
        msssim_val = float("nan")
        vif_val    = float("nan")

    # SD — standard deviation of the prediction (detail richness)
    sd_val = float(np.std(pred))

    # EN — Shannon entropy of the prediction histogram
    hist, _ = np.histogram(pred_u8.flatten(), bins=256, range=(0, 255))
    hist     = hist.astype(float) / (hist.sum() + 1e-10)
    en_val   = float(-np.sum(hist * np.log2(hist + 1e-10)))

    return {
        "PSNR":    psnr_val,
        "SSIM":    ssim_val,
        "MS-SSIM": msssim_val,
        "VIF":     vif_val,
        "SD":      sd_val,
        "EN":      en_val,
        "MSE":     mse_val,
        "MAE":     mae_val,
    }


# ============================================================================ #
#   5.  MAIN
# ============================================================================ #

def run(split: str, image_size: int, batch_size: int,
        decoder_epochs: int, skip_metrics: bool):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[✓] Device : {device}")
    print(f"[✓] Split  : {split}")

    # ── output dirs ────────────────────────────────────────────────────────── #
    out_dir = OUTPUT_ROOT / split
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # ── encoder ────────────────────────────────────────────────────────────── #
    encoder = load_encoder(CHECKPOINT, device)

    # ── dataset ────────────────────────────────────────────────────────────── #
    dataset = MRIDataset(split=split, image_size=image_size)

    # ── decoder ────────────────────────────────────────────────────────────── #
    decoder = TinyDecoder().to(device)
    if not skip_metrics:
        decoder = train_decoder(encoder, decoder, dataset, device,
                                epochs=decoder_epochs, batch_size=batch_size)

    # ── inference loop ─────────────────────────────────────────────────────── #
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    all_metrics = []
    METRIC_KEYS = ["PSNR", "SSIM", "MS-SSIM", "VIF", "SD", "EN", "MSE", "MAE"]

    print(f"[→] Extracting embeddings from {len(dataset)} slices ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Extracting"):
            enc_imgs = batch["enc"].to(device)    # (B,3,256,256)
            raw_imgs = batch["raw"]               # (B,1,256,256) on CPU
            paths    = batch["path"]
            patients = batch["patient"]

            # ── encoder forward ──────────────────────────────────────────── #
            feats = encoder(enc_imgs)             # (B,256,64,64)

            # ── save embeddings ──────────────────────────────────────────── #
            for i, (path, patient) in enumerate(zip(paths, patients)):
                # Mirror patient folder structure under embeddings/
                patient_emb_dir = emb_dir / patient
                patient_emb_dir.mkdir(exist_ok=True)

                stem    = Path(path).stem          # e.g. "000"
                npy_out = patient_emb_dir / f"{stem}.npy"
                np.save(str(npy_out), feats[i].cpu().numpy())
                # Shape saved: (256, 64, 64)

            # ── metrics (needs decoder) ──────────────────────────────────── #
            if not skip_metrics:
                recons = decoder(feats).cpu().numpy()  # (B,1,256,256)
                raws   = raw_imgs.numpy()              # (B,1,256,256)

                for i, path in enumerate(paths):
                    pred   = recons[i, 0]     # (256,256) float32
                    target = raws[i, 0]       # (256,256) float32
                    m      = compute_metrics(pred, target)
                    m["image"]   = path
                    m["patient"] = patients[i]
                    all_metrics.append(m)

    # ── print + save metrics ───────────────────────────────────────────────── #
    if all_metrics:
        print("\n" + "=" * 58)
        print(f"  RESULTS — {split.upper()} split  ({len(all_metrics)} slices)")
        print("=" * 58)
        print(f"  {'Metric':<12}  {'Mean':>10}  {'Std':>10}")
        print("-" * 58)
        summary = {}
        for k in METRIC_KEYS:
            vals = [m[k] for m in all_metrics
                    if not (isinstance(m[k], float) and np.isnan(m[k]))]
            mean_v = float(np.mean(vals)) if vals else float("nan")
            std_v  = float(np.std(vals))  if vals else float("nan")
            summary[k] = {"mean": mean_v, "std": std_v}
            print(f"  {k:<12}  {mean_v:>10.4f}  {std_v:>10.4f}")
        print("=" * 58)

        # Save CSV
        import csv
        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient", "image"] + METRIC_KEYS)
            writer.writeheader()
            writer.writerows(all_metrics)

        # Save JSON
        json_path = out_dir / "metrics_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[✓] Per-slice CSV   → {csv_path}")
        print(f"[✓] Summary JSON    → {json_path}")

    print(f"[✓] Embeddings      → {emb_dir}")
    print(f"    Shape per slice : (256, 64, 64)\n")


# ============================================================================ #
#   CLI
# ============================================================================ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedSAM2D CT Encoder — Feature Extractor + Metrics"
    )
    parser.add_argument(
        "--split", choices=["train", "val"], default="train",
        help="Which dataset split to process"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Resize CT slices to this square size (default: 256)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference (reduce to 8 if OOM)"
    )
    parser.add_argument(
        "--decoder_epochs", type=int, default=5,
        help="Epochs to train tiny decoder for metric computation"
    )
    parser.add_argument(
        "--skip_metrics", action="store_true",
        help="Skip decoder training + metrics; only save embeddings"
    )

    args = parser.parse_args()
    run(
        split          = args.split,
        image_size     = args.image_size,
        batch_size     = args.batch_size,
        decoder_epochs = args.decoder_epochs,
        skip_metrics   = args.skip_metrics,
    )