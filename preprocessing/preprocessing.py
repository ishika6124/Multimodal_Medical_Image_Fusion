import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_ROOT = "/home/teaching/group46/Mask-DiFuser-main/dataset/brain"
OUTPUT_ROOT = "/home/teaching/group46/attempt_4"
TARGET_SIZE = (256, 256)

def normalize(img, v_min, v_max):
    return np.clip((img - v_min) / (v_max - v_min + 1e-8), 0, 1)

def resize(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

def keep_slice_gradient(img, threshold=0.02):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return np.mean(grad) > threshold

patients = sorted([p for p in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, p))])
train_ids, temp_ids = train_test_split(patients, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
split_map = {"train": train_ids, "val": val_ids, "test": test_ids}

for split, patient_list in split_map.items():
    print(f"\nProcessing {split}")
    for pid in tqdm(patient_list):

        pdir = os.path.join(DATA_ROOT, pid)
        files = os.listdir(pdir)

        ct_file = next((f for f in files if "ct" in f.lower()), None)
        mr_file = next((f for f in files if "mr" in f.lower()), None)
        mask_file = next((f for f in files if "mask" in f.lower()), None)

        if not all([ct_file, mr_file, mask_file]):
            continue

        ct_vol = nib.load(os.path.join(pdir, ct_file)).get_fdata()
        mr_vol = nib.load(os.path.join(pdir, mr_file)).get_fdata()
        mask_vol = (nib.load(os.path.join(pdir, mask_file)).get_fdata() > 0).astype(np.float32)

        ct_min, ct_max = ct_vol.min(), ct_vol.max()
        mr_min, mr_max = mr_vol.min(), mr_vol.max()

        ct_out = os.path.join(OUTPUT_ROOT, split, pid, "CT")
        mr_out = os.path.join(OUTPUT_ROOT, split, pid, "MR")
        os.makedirs(ct_out, exist_ok=True)
        os.makedirs(mr_out, exist_ok=True)

        kept = 0

        for i in range(ct_vol.shape[-1]):

            ct = ct_vol[:, :, i]
            mr = mr_vol[:, :, i]
            mask = mask_vol[:, :, i]

            if np.sum(mask)/mask.size < 0.03:
                continue

            ct = normalize(ct * mask, ct_min, ct_max)
            mr = normalize(mr * mask, mr_min, mr_max)

            if not keep_slice_gradient(ct) and not keep_slice_gradient(mr):
              continue

            ct = resize(ct)
            mr = resize(mr)

            cv2.imwrite(os.path.join(ct_out, f"{kept:03d}.png"), (ct*255).astype(np.uint8))
            cv2.imwrite(os.path.join(mr_out, f"{kept:03d}.png"), (mr*255).astype(np.uint8))
            kept += 1

        print(f"{pid} → {kept}")