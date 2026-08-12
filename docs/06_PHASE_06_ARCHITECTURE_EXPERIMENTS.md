# Phase 06 — Restoration Architecture Experiments

## Objective
Find the best quality/complexity tradeoff for unified denoising + 2× super-resolution.

KLA permits CNNs, transformers, algorithm-unrolling approaches and custom/hybrid methods, but large models may reduce throughput.

## Architecture Ladder

### Model A — Small Residual CNN

```text
1×128×128
 ↓ Conv
 ↓ Residual blocks
 ↓ PixelShuffle ×2
 ↓ Conv
 ↓ 1×256×256
```

Purpose: low-cost baseline.

### Model B — Residual U-Net

```text
Input
 ↓
Encoder 32→64→128
 ↓
Bottleneck
 ↓
Decoder 128→64→32 with skip connections
 ↓
PixelShuffle ×2
 ↓
Output
```

Purpose: multi-scale context + local detail preservation.

### Model C — NAFNet-style lightweight restoration core + SR head

Use restoration-oriented blocks with:

- convolutional feature extraction
- simplified nonlinear gating
- channel attention or channel modulation
- residual learning

The exact implementation must be kept lightweight and reproducible.

### Model D — Lightweight attention hybrid

Add a small attention module only if it improves validation quality enough to justify latency.

### Future Model E — Frequency/spatial hybrid

```text
Spatial branch
   +
Frequency branch
   ↓
Feature fusion
   ↓
Restoration core
   ↓
×2 upsampling
```

Only investigate after the CNN baselines are stable.

## Common Output Head

For all architectures:

```text
feature tensor
 ↓
3×3 Conv
 ↓
PixelShuffle(upscale=2)
 ↓
3×3 Conv → 1 channel
```

PixelShuffle should be used consistently when comparing architecture cores, so the comparison is fair.

## Model Selection Criteria

Primary:

- PSNR
- SSIM
- LPIPS

Secondary:

- parameters
- peak VRAM
- end-to-end runtime
- training stability

## Controlled Experiment Rules

When comparing architecture A vs B, keep fixed:

- split
- dataset
- synthetic-data ratio
- augmentation
- loss
- optimizer if possible
- training budget
- seed policy

## Verification

For each model record:

```text
params
FLOPs or approximate compute if available
VRAM
PSNR
SSIM
LPIPS
runtime
failure cases
```

## Deliverables

```text
src/models/
├── rescnn.py
├── resunet.py
├── naf_restoration.py
└── attention_hybrid.py

results/phase06_architectures/
├── architecture_comparison.csv
├── checkpoints/
└── visuals/
```

## Acceptance Criterion

Select a candidate architecture based on joint restoration quality and throughput. A model that is marginally better but substantially slower must be justified before adoption.
