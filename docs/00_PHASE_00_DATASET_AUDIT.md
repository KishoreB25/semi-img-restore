# Phase 00 — Dataset Audit & Problem Reconnaissance

## Objective
Establish a verified, reproducible understanding of the official KLA dataset before model development.

The current reconnaissance has established:

- Training pairs: **3,200 GT + 3,200 NoisyLR**
- Hidden/evaluation test inputs currently available: **400 NoisyLR**
- GT shape: **256×256**, grayscale, `float32`, range `[0, 1]`
- NoisyLR shape: **128×128**, grayscale, `float32`
- Observed NoisyLR range: approximately `-0.05` to `1.58`
- Task: restore `128×128` degraded observations to `256×256` clean targets
- Benchmark degradations: **speckle noise, additive Gaussian noise, downsampling**, with unknown order

KLA explicitly states that values outside `[0,1]` in NoisyLR are intentional and that outputs are scored exactly as saved by the inference pipeline. [Source: KLA Problem Statement, Dataset and Evaluator requirements.]

## Scope

### In scope
- File structure and filename pairing
- Array shape/dtype/range verification
- GT↔NoisyLR correspondence
- Visual inspection and histograms
- Dataset statistics
- Train/validation partition design
- Identification of source/group information if present

### Out of scope
- Final neural architecture selection
- Full training
- Hyperparameter optimization
- Test-set model selection

## Required Features

1. Discover all `.npy` files recursively.
2. Verify GT and NoisyLR counts.
3. Verify one-to-one pairing.
4. Verify shapes and dtypes.
5. Compute dataset-wide min/max/mean/std/percentiles.
6. Generate side-by-side GT/NoisyLR visualizations.
7. Generate aligned histogram comparisons.
8. Flag malformed or unmatched samples.
9. Save an immutable dataset audit report.
10. Save a machine-readable metadata CSV/JSON.

## Expected Folder Structure

```text
project/
├── data/
│   ├── train/
│   │   ├── GT/
│   │   └── NoisyLR/
│   └── test/
│       └── NoisyLR/
├── reports/
│   └── phase00_dataset_audit/
├── src/
│   └── dataset_audit.py
└── scripts/
    └── run_dataset_audit.py
```

## Implementation Plan

### Step 1 — Discover files

Implement a file scanner that returns sorted lists of all `.npy` files.

### Step 2 — Verify pairing

Pair files by the official naming convention. Do not assume positional pairing unless the dataset specification confirms it.

For every pair:

```python
gt = np.load(gt_path)
noisy = np.load(noisy_path)
```

### Step 3 — Validate arrays

For GT:

```text
ndim   = 2
shape  = (256, 256)
dtype  = float32
range  = [0, 1]
```

For NoisyLR:

```text
ndim   = 2
shape  = (128, 128)
dtype  = float32
range  = not assumed to be [0,1]
```

### Step 4 — Compute statistics

For every image and globally calculate:

- min
- max
- mean
- std
- p01
- p05
- p50
- p95
- p99

Store results in `reports/phase00_dataset_audit/statistics.csv`.

### Step 5 — Visual inspection

Create at least 20 randomly selected paired visualizations containing:

```text
GT | NoisyLR | GT histogram | NoisyLR histogram
```

### Step 6 — Save audit manifest

Create:

```text
reports/phase00_dataset_audit/manifest.json
reports/phase00_dataset_audit/statistics.csv
reports/phase00_dataset_audit/sample_pairs.png
```

## Normalization Policy

### GT
GT is already normalized to `[0,1]`.

### NoisyLR
Do **not**:

- clip the input
- apply per-image min-max normalization
- independently rescale each image to `[0,1]`

The baseline pipeline must preserve the raw NoisyLR float32 values.

Any alternative normalization experiment must be explicitly named and evaluated as an experiment, never silently introduced into the production pipeline.

## Verification Checklist

- [ ] Exactly 3,200 train GT samples found.
- [ ] Exactly 3,200 train NoisyLR samples found.
- [ ] Exactly 400 test NoisyLR samples found.
- [ ] All training pairs matched.
- [ ] All GT arrays are `(256,256)` and `float32`.
- [ ] All NoisyLR arrays are `(128,128)` and `float32`.
- [ ] GT range is within `[0,1]`.
- [ ] NoisyLR out-of-range values are preserved.
- [ ] No NaN or Inf values.
- [ ] Visualization confirms correct pairing.
- [ ] Audit report is reproducible.

## Deliverables

```text
reports/phase00_dataset_audit/manifest.json
reports/phase00_dataset_audit/statistics.csv
reports/phase00_dataset_audit/sample_pairs.png
reports/phase00_dataset_audit/audit_summary.md
```

## Testing

### Unit tests

1. Valid GT array accepted.
2. Wrong GT shape rejected.
3. Wrong dtype rejected or explicitly converted.
4. NaN/Inf detected.
5. Missing pair detected.
6. Input values outside `[0,1]` are not treated as an error for NoisyLR.

### Acceptance criterion

No model development proceeds until the dataset audit passes with zero unexplained pairing or shape errors.
