# Phase 01 — Data Pipeline, Bicubic Baseline & Project Skeleton

## Objective
Build the first complete data-to-output path and establish a non-neural baseline.

This phase must answer:

> Can the team reliably load a paired `.npy` sample, produce the expected 256×256 output, save it in the correct format, and calculate restoration metrics?

## Scope

### In scope
- PyTorch Dataset/DataLoader
- deterministic train/validation indexing
- bicubic ×2 baseline
- PSNR/SSIM/LPIPS evaluation pipeline
- experiment logging
- project repository structure

### Out of scope
- sophisticated neural architectures
- final loss design
- extensive synthetic degradation

## Folder Structure

```text
project/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── configs/
│   └── baseline.yaml
├── src/
│   ├── dataset.py
│   ├── baseline.py
│   ├── metrics.py
│   ├── io.py
│   └── utils.py
├── weights/
├── results/
│   └── phase01_baseline/
└── tests/
    ├── test_dataset.py
    ├── test_metrics.py
    └── test_io.py
```

## Data Pipeline

```text
.npy file
   ↓
np.load
   ↓
float32
   ↓
shape [1,H,W]
   ↓
PyTorch Tensor
   ↓
DataLoader
```

Do not alter NoisyLR values at this stage.

## Baseline Model

Use classical bicubic interpolation:

\[
\hat{x}_{bicubic} = Bicubic(y, scale=2)
\]

where:

- `y` = NoisyLR `128×128`
- output = `256×256`

This baseline does **not** denoise the image. That is intentional: it provides the simplest reference against which learned restoration must improve.

## Metrics

Implement:

### PSNR

\[
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
\]

For GT normalized to `[0,1]`, use `MAX_I = 1`.

\[
MSE = \frac{1}{N}\sum_i(x_i-\hat{x}_i)^2
\]

### SSIM

Compute on the restored grayscale image and GT.

### LPIPS

Use a reproducible grayscale-compatible evaluation procedure. If a library expects 3 channels, replicate the grayscale channel consistently for evaluation only:

```text
1-channel → [1,1,1] → LPIPS
```

Document the exact LPIPS model/version used.

## Post-processing

For baseline and later model outputs:

```python
output = np.clip(output, 0.0, 1.0)
```

Do not clip or normalize the input.

## Implementation Plan

### Step 1
Create a deterministic pair manifest.

### Step 2
Create `RestorationDataset`.

### Step 3
Implement a seeded train/validation split.

### Step 4
Implement bicubic inference.

### Step 5
Calculate PSNR/SSIM/LPIPS.

### Step 6
Save baseline predictions.

### Step 7
Create `experiments.csv` with:

```text
experiment_id, model, loss, augmentation, synthetic_data,
psnr, ssim, lpips, runtime_ms_per_image, notes
```

## Verification

For one sample:

```text
input shape  = (1,128,128)
output shape = (1,256,256)
```

Check:

- deterministic output
- no shape mismatch
- no NaN/Inf
- output in `[0,1]` after clipping
- metrics reproduce on repeated runs

## Deliverables

```text
src/dataset.py
src/baseline.py
src/metrics.py
results/phase01_baseline/metrics.json
results/phase01_baseline/visuals/
configs/baseline.yaml
```

## Testing

### Unit tests
- pair loading
- image shape
- dtype conversion
- deterministic indexing
- bicubic output size
- metrics with identical images

### Acceptance criterion

The baseline pipeline must execute from a clean shell command without notebook-only state.
