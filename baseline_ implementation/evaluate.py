# # import os
# # import cv2
# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm
# # from skimage.metrics import structural_similarity as ssim
# # from skimage.measure import shannon_entropy
# # from sewar.full_ref import vifp
# # import torch
# # import pytorch_msssim

# # # Paths
# # ROOT = "/home/teaching/group46/attempt_4/val"
# # SAVE_PATH = "/home/teaching/group46/attempt_4/metrics.csv"

# # # ── Metrics ─────────────────────────────────────────

# # def compute_ssim(a, b):
# #     return ssim(a, b, data_range=255)

# # def compute_msssim(a, b):
# #     a = torch.tensor(a/255.).unsqueeze(0).unsqueeze(0).float()
# #     b = torch.tensor(b/255.).unsqueeze(0).unsqueeze(0).float()
# #     return pytorch_msssim.ms_ssim(a, b, data_range=1.).item()

# # def compute_sd(img):
# #     return np.std(img)

# # def compute_en(img):
# #     return shannon_entropy(img)

# # def compute_vif(a, b):
# #     return vifp(a, b)

# # # ── Load existing results (for resume) ───────────────

# # if os.path.exists(SAVE_PATH):
# #     df_existing = pd.read_csv(SAVE_PATH)
# #     done_patients = set(df_existing["patient"].values)
# # else:
# #     df_existing = pd.DataFrame()
# #     done_patients = set()

# # results = []

# # patients = sorted(os.listdir(ROOT))

# # for pid in tqdm(patients):

# #     if pid in done_patients:
# #         print(f"⏭ Skipping {pid} (already evaluated)")
# #         continue

# #     ct_dir    = os.path.join(ROOT, pid, "CT")
# #     mr_dir    = os.path.join(ROOT, pid, "MR")
# #     fused_dir = os.path.join(ROOT, pid, "Fused")

# #     if not os.path.exists(fused_dir):
# #         continue

# #     files = sorted(os.listdir(fused_dir))

# #     ssim_list, mssim_list, vif_list, sd_list, en_list = [], [], [], [], []

# #     for f in files:

# #         ct_path    = os.path.join(ct_dir, f)
# #         mr_path    = os.path.join(mr_dir, f)
# #         fused_path = os.path.join(fused_dir, f)

# #         if not (os.path.exists(ct_path) and os.path.exists(mr_path)):
# #             continue

# #         ct    = cv2.imread(ct_path, 0)
# #         mr    = cv2.imread(mr_path, 0)
# #         fused = cv2.imread(fused_path, 0)

# #         # Metrics
# #         ssim_val  = (compute_ssim(fused, ct) + compute_ssim(fused, mr)) / 2
# #         mssim_val = (compute_msssim(fused, ct) + compute_msssim(fused, mr)) / 2
# #         vif_val   = (compute_vif(fused, ct) + compute_vif(fused, mr)) / 2
# #         sd_val    = compute_sd(fused)
# #         en_val    = compute_en(fused)

# #         ssim_list.append(ssim_val)
# #         mssim_list.append(mssim_val)
# #         vif_list.append(vif_val)
# #         sd_list.append(sd_val)
# #         en_list.append(en_val)

# #     if len(ssim_list) == 0:
# #         continue

# #     row = {
# #         "patient": pid,
# #         "SSIM": np.mean(ssim_list),
# #         "MS-SSIM": np.mean(mssim_list),
# #         "VIF": np.mean(vif_list),
# #         "SD": np.mean(sd_list),
# #         "EN": np.mean(en_list)
# #     }

# #     results.append(row)

# #     # ✅ Save incrementally (important for resume)
# #     df_temp = pd.DataFrame([row])
# #     df_temp.to_csv(SAVE_PATH, mode='a', header=not os.path.exists(SAVE_PATH), index=False)

# # # ── Final overall average ────────────────────────────

# # df = pd.read_csv(SAVE_PATH)

# # overall = df.mean(numeric_only=True)
# # overall["patient"] = "OVERALL"

# # df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
# # df.to_csv(SAVE_PATH, index=False)

# # print("\n✅ Evaluation completed (with resume support)")
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from skimage.measure import shannon_entropy

# import torch
# import torch.nn.functional as F
# from pytorch_msssim import ssim, ms_ssim
# from piq import vif_p   # same library as training code

# # ── Paths ────────────────────────────────────────────
# ROOT      = "/home/teaching/group46/attempt_4/test"
# SAVE_PATH = "/home/teaching/group46/attempt_4/metrics.csv"

# # ── Metric functions — exactly matching training compute_metrics() ────────

# def to_tensor(img_uint8):
#     """uint8 numpy (H,W) → float32 tensor (1,1,H,W) in [0,1]"""
#     return torch.tensor(img_uint8 / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# def compute_psnr(p, r):
#     """p, r: (1,1,H,W) tensors in [0,1]. Matches training formula exactly."""
#     mse = ((p - r) ** 2).mean(dim=[1, 2, 3])
#     return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()

# def compute_ssim(p, r):
#     return ssim(p, r, data_range=1.0, size_average=True).item()

# def compute_msssim(p, r):
#     return ms_ssim(p, r, data_range=1.0, size_average=True).item()

# def compute_vif(p, r):
#     return vif_p(p, r, data_range=1.0).item()

# def compute_mse(p, r):
#     return F.mse_loss(p, r).item()

# def compute_mae(p, r):
#     return F.l1_loss(p, r).item()

# def compute_sd(img_uint8):
#     """Standard deviation on fused image — no reference needed."""
#     return float(np.std(img_uint8 / 255.))

# def compute_en(img_uint8):
#     """Shannon entropy on fused image — no reference needed."""
#     arr = img_uint8 / 255.
#     hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=True)
#     hist     = hist + 1e-10
#     hist    /= hist.sum()
#     return float(-(hist * np.log2(hist)).sum())

# def compute_overall(psnr_v, ssim_v, msssim_v, vif_v):
#     """Matches training: (psnr/50 + ssim + ms_ssim + vif) / 4"""
#     return (psnr_v / 50.0 + ssim_v + msssim_v + vif_v) / 4.0

# # ── Resume support ────────────────────────────────────
# if os.path.exists(SAVE_PATH):
#     df_existing   = pd.read_csv(SAVE_PATH)
#     df_existing   = df_existing[df_existing["patient"] != "OVERALL"]
#     done_patients = set(df_existing["patient"].values)
# else:
#     done_patients = set()

# COLUMNS = ["patient", "PSNR", "SSIM", "MS-SSIM", "VIF",
#            "SD", "EN", "MSE", "MAE", "Overall"]

# patients = sorted(os.listdir(ROOT))

# for pid in tqdm(patients):

#     if pid in done_patients:
#         print(f"⏭  Skipping {pid} (already evaluated)")
#         continue

#     ct_dir    = os.path.join(ROOT, pid, "CT")
#     mr_dir    = os.path.join(ROOT, pid, "MR")
#     fused_dir = os.path.join(ROOT, pid, "Fused")

#     if not os.path.exists(fused_dir):
#         print(f"⚠  No Fused folder for {pid}, skipping.")
#         continue

#     files = sorted(os.listdir(fused_dir))

#     psnr_list, ssim_list, msssim_list, vif_list = [], [], [], []
#     mse_list, mae_list, sd_list, en_list        = [], [], [], []

#     for f in files:
#         ct_path    = os.path.join(ct_dir,    f)
#         mr_path    = os.path.join(mr_dir,    f)
#         fused_path = os.path.join(fused_dir, f)

#         if not (os.path.exists(ct_path) and os.path.exists(mr_path)):
#             continue

#         ct_np    = cv2.imread(ct_path,    cv2.IMREAD_GRAYSCALE)
#         mr_np    = cv2.imread(mr_path,    cv2.IMREAD_GRAYSCALE)
#         fused_np = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)

#         if ct_np is None or mr_np is None or fused_np is None:
#             print(f"  ⚠  Could not read {f}, skipping.")
#             continue

#         # ── Convert to [0,1] tensors ──────────────────────────────────────
#         p = to_tensor(fused_np)                     # (1,1,H,W) predicted
#         c = to_tensor(ct_np)
#         m = to_tensor(mr_np)

#         # ── Reference = mean of CT and MRI — EXACTLY as in training ───────
#         r = (c + m) * 0.5                           # (1,1,H,W)

#         p = p.clamp(0, 1)
#         r = r.clamp(0, 1)

#         psnr_v   = compute_psnr(p, r)
#         ssim_v   = compute_ssim(p, r)
#         msssim_v = compute_msssim(p, r)
#         vif_v    = compute_vif(p, r)
#         mse_v    = compute_mse(p, r)
#         mae_v    = compute_mae(p, r)
#         sd_v     = compute_sd(fused_np)
#         en_v     = compute_en(fused_np)

#         psnr_list.append(psnr_v)
#         ssim_list.append(ssim_v)
#         msssim_list.append(msssim_v)
#         vif_list.append(vif_v)
#         mse_list.append(mse_v)
#         mae_list.append(mae_v)
#         sd_list.append(sd_v)
#         en_list.append(en_v)

#     if len(ssim_list) == 0:
#         print(f"  ⚠  No valid slices for {pid}, skipping.")
#         continue

#     # ── Patient-level mean across all slices ─────────────────────────────
#     mean_psnr    = np.mean(psnr_list)
#     mean_ssim    = np.mean(ssim_list)
#     mean_msssim  = np.mean(msssim_list)
#     mean_vif     = np.mean(vif_list)
#     mean_mse     = np.mean(mse_list)
#     mean_mae     = np.mean(mae_list)
#     mean_sd      = np.mean(sd_list)
#     mean_en      = np.mean(en_list)
#     mean_overall = compute_overall(mean_psnr, mean_ssim, mean_msssim, mean_vif)

#     row = {
#         "patient": pid,
#         "PSNR":    round(mean_psnr,    6),
#         "SSIM":    round(mean_ssim,    6),
#         "MS-SSIM": round(mean_msssim,  6),
#         "VIF":     round(mean_vif,     6),
#         "SD":      round(mean_sd,      6),
#         "EN":      round(mean_en,      6),
#         "MSE":     round(mean_mse,     6),
#         "MAE":     round(mean_mae,     6),
#         "Overall": round(mean_overall, 6),
#     }

#     # ── Incremental save ──────────────────────────────────────────────────
#     write_header = not os.path.exists(SAVE_PATH)
#     pd.DataFrame([row]).to_csv(SAVE_PATH, mode='a', header=write_header, index=False)

#     print(f"  ✅ {pid:12s} | PSNR={mean_psnr:5.2f} | SSIM={mean_ssim:.4f} | "
#           f"MS-SSIM={mean_msssim:.4f} | VIF={mean_vif:.4f} | "
#           f"MSE={mean_mse:.5f} | MAE={mean_mae:.5f} | Overall={mean_overall:.4f}")

# # ── Final OVERALL row ─────────────────────────────────────────────────────
# df = pd.read_csv(SAVE_PATH)
# df = df[df["patient"] != "OVERALL"]   # remove any stale row

# overall_row            = df.mean(numeric_only=True).round(6).to_dict()
# overall_row["patient"] = "OVERALL"

# df_final = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
# df_final.to_csv(SAVE_PATH, index=False)

# # ── Print final summary ───────────────────────────────────────────────────
# print("\n" + "="*90)
# print(f"{'Patient':<12} {'PSNR':>6} {'SSIM':>7} {'MS-SSIM':>8} {'VIF':>7} "
#       f"{'SD':>7} {'EN':>7} {'MSE':>8} {'MAE':>8} {'Overall':>8}")
# print("="*90)
# for _, r in df_final.iterrows():
#     print(f"{str(r['patient']):<12} {r['PSNR']:>6.3f} {r['SSIM']:>7.4f} "
#           f"{r['MS-SSIM']:>8.4f} {r['VIF']:>7.4f} {r['SD']:>7.4f} "
#           f"{r['EN']:>7.4f} {r['MSE']:>8.5f} {r['MAE']:>8.5f} {r['Overall']:>8.4f}")
# print("="*90)
# print(f"\n✅ Saved → {SAVE_PATH}")
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from piq import vif_p

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = "/home/teaching/group46/attempt_4/test"
SAVE_PATH = "/home/teaching/group46/attempt_4/metrics.csv"

# ── Metric helpers — match inference scripts exactly ──────────────────────────

def to_tensor(img_uint8):
    """uint8 numpy (H,W) → float32 tensor (1,1,H,W) in [0,1]"""
    return torch.tensor(img_uint8 / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def compute_psnr(p, r):
    mse = ((p - r) ** 2).mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()

def compute_sd(img_uint8):
    return float(np.std(img_uint8 / 255.))

def compute_en(img_uint8):
    arr  = img_uint8 / 255.
    hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=False)
    hist = hist.astype(np.float64) + 1e-10
    hist /= hist.sum()
    return float(-(hist * np.log2(hist)).sum())

def compute_overall(psnr_mean, ssim_mean, msssim_mean, vif_mean):
    """Matches inference: (psnr/50 + ssim + ms_ssim + vif) / 4"""
    return (psnr_mean / 50.0 + ssim_mean + msssim_mean + vif_mean) / 4.0

# ── Columns — mirrors inference metrics.csv exactly ───────────────────────────
COLUMNS = [
    "patient",
    # vs CT
    "psnr_ct", "ssim_ct", "msssim_ct",
    # vs MRI
    "psnr_mri", "ssim_mri", "msssim_mri",
    # vs Mean
    "psnr_mean", "ssim_mean", "msssim_mean",
    "vif", "mse", "mae",
    # Image quality
    "sd", "en", "overall",
]

# ── Resume support ────────────────────────────────────────────────────────────
if os.path.exists(SAVE_PATH):
    df_existing   = pd.read_csv(SAVE_PATH)
    df_existing   = df_existing[df_existing["patient"] != "OVERALL"]
    done_patients = set(df_existing["patient"].values)
else:
    done_patients = set()

patients = sorted(os.listdir(ROOT))

for pid in tqdm(patients):

    if pid in done_patients:
        print(f"⏭  Skipping {pid} (already evaluated)")
        continue

    ct_dir    = os.path.join(ROOT, pid, "CT")
    mr_dir    = os.path.join(ROOT, pid, "MR")
    fused_dir = os.path.join(ROOT, pid, "Fused")

    if not os.path.exists(fused_dir):
        print(f"⚠  No Fused folder for {pid}, skipping.")
        continue

    files = sorted(os.listdir(fused_dir))

    # Per-slice accumulators — one list per metric, matching inference fieldnames
    psnr_ct_list,   ssim_ct_list,   msssim_ct_list   = [], [], []
    psnr_mri_list,  ssim_mri_list,  msssim_mri_list  = [], [], []
    psnr_mean_list, ssim_mean_list, msssim_mean_list  = [], [], []
    vif_list, mse_list, mae_list, sd_list, en_list    = [], [], [], [], []

    for f in files:
        ct_path    = os.path.join(ct_dir,    f)
        mr_path    = os.path.join(mr_dir,    f)
        fused_path = os.path.join(fused_dir, f)

        if not (os.path.exists(ct_path) and os.path.exists(mr_path)):
            continue

        ct_np    = cv2.imread(ct_path,    cv2.IMREAD_GRAYSCALE)
        mr_np    = cv2.imread(mr_path,    cv2.IMREAD_GRAYSCALE)
        fused_np = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)

        if ct_np is None or mr_np is None or fused_np is None:
            print(f"  ⚠  Could not read {f}, skipping.")
            continue

        # ── Tensors — (1,1,H,W) in [0,1] ────────────────────────────────────
        p = to_tensor(fused_np).clamp(0, 1)
        c = to_tensor(ct_np).clamp(0, 1)
        m = to_tensor(mr_np).clamp(0, 1)
        r = (0.5 * (c + m)).clamp(0, 1)   # mean reference, exactly as in inference

        # ── vs CT ─────────────────────────────────────────────────────────────
        psnr_ct_list.append(compute_psnr(p, c))
        ssim_ct_list.append(ssim(p, c, data_range=1., size_average=True).item())
        msssim_ct_list.append(ms_ssim(p, c, data_range=1., size_average=True).item())

        # ── vs MRI ────────────────────────────────────────────────────────────
        psnr_mri_list.append(compute_psnr(p, m))
        ssim_mri_list.append(ssim(p, m, data_range=1., size_average=True).item())
        msssim_mri_list.append(ms_ssim(p, m, data_range=1., size_average=True).item())

        # ── vs Mean ───────────────────────────────────────────────────────────
        psnr_mean_list.append(compute_psnr(p, r))
        ssim_mean_list.append(ssim(p, r, data_range=1., size_average=True).item())
        msssim_mean_list.append(ms_ssim(p, r, data_range=1., size_average=True).item())
        vif_list.append(vif_p(p, r, data_range=1.).item())
        mse_list.append(F.mse_loss(p, r).item())
        mae_list.append(F.l1_loss(p, r).item())

        # ── Image quality (no reference) ──────────────────────────────────────
        sd_list.append(compute_sd(fused_np))
        en_list.append(compute_en(fused_np))

    if len(ssim_ct_list) == 0:
        print(f"  ⚠  No valid slices for {pid}, skipping.")
        continue

    # ── Patient-level mean across all slices ──────────────────────────────────
    mean_psnr_ct    = np.mean(psnr_ct_list)
    mean_ssim_ct    = np.mean(ssim_ct_list)
    mean_msssim_ct  = np.mean(msssim_ct_list)

    mean_psnr_mri   = np.mean(psnr_mri_list)
    mean_ssim_mri   = np.mean(ssim_mri_list)
    mean_msssim_mri = np.mean(msssim_mri_list)

    mean_psnr_mean   = np.mean(psnr_mean_list)
    mean_ssim_mean   = np.mean(ssim_mean_list)
    mean_msssim_mean = np.mean(msssim_mean_list)
    mean_vif         = np.mean(vif_list)
    mean_mse         = np.mean(mse_list)
    mean_mae         = np.mean(mae_list)
    mean_sd          = np.mean(sd_list)
    mean_en          = np.mean(en_list)

    # Overall uses mean-reference metrics — exactly matching inference
    mean_overall = compute_overall(mean_psnr_mean, mean_ssim_mean,
                                   mean_msssim_mean, mean_vif)

    row = {
        "patient":    pid,
        # vs CT
        "psnr_ct":    round(mean_psnr_ct,    6),
        "ssim_ct":    round(mean_ssim_ct,    6),
        "msssim_ct":  round(mean_msssim_ct,  6),
        # vs MRI
        "psnr_mri":   round(mean_psnr_mri,   6),
        "ssim_mri":   round(mean_ssim_mri,   6),
        "msssim_mri": round(mean_msssim_mri, 6),
        # vs Mean
        "psnr_mean":   round(mean_psnr_mean,   6),
        "ssim_mean":   round(mean_ssim_mean,   6),
        "msssim_mean": round(mean_msssim_mean, 6),
        "vif":         round(mean_vif,         6),
        "mse":         round(mean_mse,         6),
        "mae":         round(mean_mae,         6),
        # Image quality
        "sd":          round(mean_sd,          6),
        "en":          round(mean_en,          6),
        "overall":     round(mean_overall,     6),
    }

    # ── Incremental save ──────────────────────────────────────────────────────
    write_header = not os.path.exists(SAVE_PATH)
    pd.DataFrame([row]).to_csv(SAVE_PATH, mode='a', header=write_header, index=False)

    print(f"  ✅ {pid:12s} | "
          f"PSNR_CT={mean_psnr_ct:5.2f} | SSIM_CT={mean_ssim_ct:.4f} | "
          f"PSNR_MRI={mean_psnr_mri:5.2f} | SSIM_MRI={mean_ssim_mri:.4f} | "
          f"PSNR_mean={mean_psnr_mean:5.2f} | SSIM_mean={mean_ssim_mean:.4f} | "
          f"VIF={mean_vif:.4f} | MSE={mean_mse:.5f} | MAE={mean_mae:.5f} | "
          f"Overall={mean_overall:.4f}")

# ── Final OVERALL row ─────────────────────────────────────────────────────────
df = pd.read_csv(SAVE_PATH)
df = df[df["patient"] != "OVERALL"]   # remove any stale OVERALL row

overall_row            = df.mean(numeric_only=True).round(6).to_dict()
overall_row["patient"] = "OVERALL"

df_final = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
df_final.to_csv(SAVE_PATH, index=False)

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'Patient':<12} {'PSNR_CT':>8} {'SSIM_CT':>8} {'PSNR_MRI':>9} {'SSIM_MRI':>9} "
      f"{'PSNR_mean':>10} {'SSIM_mean':>10} {'VIF':>7} "
      f"{'MSE':>8} {'MAE':>8} {'SD':>7} {'EN':>7} {'Overall':>8}")
print("=" * 110)
for _, r in df_final.iterrows():
    print(f"{str(r['patient']):<12} "
          f"{r['psnr_ct']:>8.3f} {r['ssim_ct']:>8.4f} "
          f"{r['psnr_mri']:>9.3f} {r['ssim_mri']:>9.4f} "
          f"{r['psnr_mean']:>10.3f} {r['ssim_mean']:>10.4f} "
          f"{r['vif']:>7.4f} {r['mse']:>8.5f} {r['mae']:>8.5f} "
          f"{r['sd']:>7.4f} {r['en']:>7.4f} {r['overall']:>8.4f}")
print("=" * 110)
print(f"\n✅ Saved → {SAVE_PATH}")