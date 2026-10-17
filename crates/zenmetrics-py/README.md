# zenmetrics-py

High-performance perceptual image quality assessment in Rust with Pythonic ergonomics and first-class Jupyter Notebook integration.

## Key Features

- **Blazing Fast**: Native Rust metric implementations (SSIMULACRA 2, Butteraugli, CVVDP, DSSIM, IW-SSIM, PSNR, GMSD, HaarPSI, VIF, VMAF, ZenSIM) with zero Python VM overhead during computation.
- **Universal Input**: Effortlessly scores file paths (`.png`, `.jpg`, `.webp`, `.jxl`, `.avif`), in-memory NumPy arrays (`uint8` or normalized `float32`, `HWC` or `CHW`), PIL Images, and PyTorch tensors.
- **First-Class Jupyter Notebook Integration**: Rich `_repr_html_()` displays for scores, multi-variant rate-distortion grids, folder summaries, and ITU-T statistical panels.
- **Comparison & Rate-Distortion Grid**: Easily evaluate multiple distortion variants across multiple metrics with automatic tabular summaries and DataFrame exports.
- **Directory Batch Evaluation**: Built-in `score_folder` automatically discovers and pairs corresponding images across folder trees (even across format extensions).
- **Subjective Benchmark Stats (`zenmetrics.stats`)**: Full ITU-T P.1401 evaluation panel (SROCC, PLCC, KROCC, PWRC-AUC, Outlier Ratio, Z-RMSE, 4-parameter logistic curve fitting).

---

## Quickstart

### 1. Basic Scoring

```python
import zenmetrics as zm

# Score files
score = zm.score("reference.png", "distorted.webp", metric="ssim2")
print(score)          # ScoreResult(metric='ssim2', value=82.45, scores={'ssim2': 82.45})
print(float(score))   # 82.45
if score > 80:
    print("Passes high perceptual quality bar!")

# Score in-memory NumPy arrays or PyTorch tensors
import numpy as np
ref_arr = np.zeros((512, 512, 3), dtype=np.uint8)
dist_arr = np.full((512, 512, 3), 5, dtype=np.uint8)

res = zm.score(ref_arr, dist_arr, metric="psnr")
print(f"PSNR: {res:.2f} dB")
```

### 2. Multi-Variant Model Comparison

```python
# Compare multiple compression levels or model outputs
comp = zm.compare(
    reference="original.png",
    variants={
        "q50": "dist_q50.webp",
        "q75": "dist_q75.webp",
        "q90": "dist_q90.webp",
    },
    metrics=["ssim2", "butteraugli", "psnr"]
)

# Export to Pandas DataFrame
df = comp.to_dataframe()

# In Jupyter, `comp` displays an interactive HTML table with best values highlighted!
comp
```

### 3. Folder-to-Folder Batch Evaluation

```python
# Evaluate entire directories with smart pairing across extensions
res = zm.score_folder(
    reference_dir="dataset/ground_truth",
    distorted_dir="dataset/model_predictions",
    metric="ssim2",
    recursive=True
)

print(f"Evaluated {res.successful_pairs}/{res.total_pairs} pairs")
print(f"Mean score: {res.summary['ssim2']['mean']:.2f}")

# Convert per-image results to DataFrame
df = res.to_dataframe()
```

### 4. Subjective Quality Validation (`zenmetrics.stats`)

```python
import zenmetrics.stats as zms

# Compute ITU-T P.1401 benchmark correlation panel against human MOS scores
panel = zms.compute_panel(metric_predictions, human_mos_targets)

print(f"SROCC: {panel.srocc:.4f}")
print(f"PLCC:  {panel.plcc:.4f}")
print(f"PWRC:  {panel.pwrc_auc:.4f}")

# In Jupyter, `panel` renders an ITU-T benchmark score card!
panel
```

### 5. Callable Metric (PyIQA Compatible)

```python
butter = zm.create_metric("butteraugli")
score = butter(ref, dist)
```

---

## Available Metrics

Call `zm.list_metrics()` to list all available metrics:
- `ssim2` (SSIMULACRA 2)
- `butteraugli` (Google Butteraugli psychovisual distance)
- `cvvdp` (Color Visual Difference Predictor)
- `dssim` (Structural dissimilarity)
- `iwssim` (Information-theoretic Weighted SSIM)
- `psnr`, `psnr-y`, `psnr-y601`, `psnr-y-libvmaf`, `psnrhvs`
- `gmsd`, `mdsi`, `ms-gmsd`
- `haarpsi`
- `fsim`
- `vif`, `vifvec`
- `vmaf`, `vmaf-neg`, `vmaf-4k`
- `zensim` (ZenSIM Learned Perceptual Quality Score)
- ... and GPU-accelerated variants (`ssim2-gpu`, `cvvdp-gpu`, `iwssim-gpu`, `dssim-gpu`).
