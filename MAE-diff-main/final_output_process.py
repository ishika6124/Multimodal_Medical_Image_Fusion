import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
CSV_PATH = "/home/teaching/group46/MAEDiff-main/fusion_v2_output/val_metrics.csv"
OUT_DIR = "/home/teaching/group46/MAEDiff-main/fusion_v2_output/final_results"
Path(OUT_DIR).mkdir(exist_ok=True)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("\nLoaded CSV with shape:", df.shape)
print(df.head())

# =========================
# CLEAN DATA
# =========================
# Convert all numeric columns
for col in df.columns:
    if col != "patient":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# =========================
# OVERALL METRICS
# =========================
metrics = ["psnr", "ssim", "ms_ssim", "vif", "sd", "en", "mse", "mae", "overall"]

summary = {}

print("\n===== OVERALL METRICS =====")
for m in metrics:
    mean = df[m].mean()
    std = df[m].std()
    summary[m] = {"mean": mean, "std": std}
    print(f"{m.upper():10s} : {mean:.4f} ± {std:.4f}")

# Save summary
summary_df = pd.DataFrame(summary).T
summary_df.to_csv(f"{OUT_DIR}/summary.csv")

# =========================
# BEST & WORST PATIENTS
# =========================
best = df.loc[df["overall"].idxmax()]
worst = df.loc[df["overall"].idxmin()]

print("\n===== BEST PATIENT =====")
print(best)

print("\n===== WORST PATIENT =====")
print(worst)

# Save them
best.to_frame().T.to_csv(f"{OUT_DIR}/best_patient.csv", index=False)
worst.to_frame().T.to_csv(f"{OUT_DIR}/worst_patient.csv", index=False)

# =========================
# TOP 10 PATIENTS
# =========================
top10 = df.sort_values("overall", ascending=False).head(10)
top10.to_csv(f"{OUT_DIR}/top10.csv", index=False)

# =========================
# METRIC DISTRIBUTION PLOTS
# =========================
for m in metrics:
    plt.figure()
    plt.hist(df[m], bins=30)
    plt.title(f"{m.upper()} Distribution")
    plt.xlabel(m)
    plt.ylabel("Frequency")
    plt.savefig(f"{OUT_DIR}/{m}_hist.png")
    plt.close()

# =========================
# CORRELATION MATRIX
# =========================
corr = df[metrics].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(metrics)), metrics, rotation=45)
plt.yticks(range(len(metrics)), metrics)
plt.title("Metric Correlation")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation.png")
plt.close()

# =========================
# RANK PATIENTS
# =========================
df["rank"] = df["overall"].rank(ascending=False)

df_sorted = df.sort_values("rank")
df_sorted.to_csv(f"{OUT_DIR}/ranked_patients.csv", index=False)

# =========================
# FINAL REPORT
# =========================
with open(f"{OUT_DIR}/report.txt", "w") as f:
    f.write("===== FINAL RESULTS =====\n\n")
    for m in metrics:
        f.write(f"{m.upper()} : {summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}\n")

    f.write("\nBEST PATIENT:\n")
    f.write(str(best) + "\n")

    f.write("\nWORST PATIENT:\n")
    f.write(str(worst) + "\n")

print("\n✅ All results saved in:", OUT_DIR)