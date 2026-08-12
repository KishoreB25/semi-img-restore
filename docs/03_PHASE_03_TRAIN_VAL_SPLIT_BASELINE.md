# Phase 03 — Clean Train/Validation Split & First Neural Baseline

## Objective
Create a leakage-safe training/validation setup and obtain the first meaningful neural baseline.

## Scope

### In scope
- fixed train/validation split
- reproducible seeds
- small ResUNet-style model
- L1 loss
- baseline metrics and visuals

### Out of scope
- synthetic data
- sophisticated losses
- architecture search

## Split Strategy

Start with approximately:

```text
Train      ≈ 90%
Validation ≈ 10%
```

For 3,200 pairs this is approximately:

```text
2,880 train
320 validation
```

If the dataset contains source/group metadata, prefer a grouped split that better simulates KLA's unfamiliar-content evaluation. Do not manufacture group information that does not exist.

Freeze the split after creation.

## Folder Structure

```text
configs/
├── split.yaml
└── baseline_resunet.yaml

results/phase03_neural_baseline/
├── metrics.csv
├── curves/
├── visuals/
└── checkpoints/
```

## Model — ResUNet Baseline

```text
Input: 1×128×128
      ↓
Stem Conv 3×3 → 32
      ↓
Encoder Block 1 → 64
      ↓
Encoder Block 2 → 128
      ↓
Bottleneck → 128
      ↓
Decoder Block 2 → 64 + skip
      ↓
Decoder Block 1 → 32 + skip
      ↓
Feature Conv
      ↓
PixelShuffle ×2
      ↓
Output Conv 1 channel
```

## Layer Design

### Encoder block

```text
Conv 3×3
ReLU
Conv 3×3
Residual connection
```

### Downsampling

Use stride-2 convolution or a controlled pooling/convolution block.

### Decoder

Use interpolation/convolution or transpose convolution for feature-map upsampling inside the encoder-decoder. The final ×2 output upsampling should be learned with PixelShuffle.

## Loss

Start with:

\[
L = L_1
\]

Later phases will replace/extend this loss.

## Training

Recommended starting configuration:

```text
optimizer: AdamW
initial_lr: 2e-4
scheduler: cosine decay or plateau-based; log the choice
batch_size: determined by GPU memory
epochs: enough for stable convergence
seed: fixed
```

Do not compare experiments with different preprocessing and different models simultaneously.

## Verification

### Data
- validation samples never seen during optimization
- split manifest saved

### Model
- output is exactly `1×256×256`
- no NaN/Inf

### Metrics
Evaluate every validation epoch or at a fixed interval:

- PSNR
- SSIM
- LPIPS

### Visuals
Save:

```text
NoisyLR → Prediction → GT → Residual
```

## Deliverables

```text
configs/split.yaml
configs/baseline_resunet.yaml
results/phase03_neural_baseline/metrics.csv
results/phase03_neural_baseline/best.pth
results/phase03_neural_baseline/visuals/
```

## Acceptance Criterion

The neural baseline should be compared directly against bicubic on the same frozen validation set.
