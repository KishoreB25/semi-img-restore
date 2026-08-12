# Phase 02 — One/Two-Pair Overfit Sanity Check

## Objective
Prove that the complete training pipeline is mathematically and programmatically correct before full training.

The model must intentionally overfit one or two training pairs.

This phase is not about generalization. It is a debugging gate.

## Scope

### In scope
- tiny restoration model
- single-pair/two-pair training
- loss convergence
- visual inspection
- gradient-flow verification

### Out of scope
- augmentation
- synthetic data
- OOD evaluation
- final architecture

## Model

Use the smallest viable learnable model:

```text
Input 1×128×128
      ↓
3×3 Conv, 32 channels
      ↓
Residual Block ×4
      ↓
3×3 Conv
      ↓
PixelShuffle ×2
      ↓
3×3 Conv, 1 channel
      ↓
Output 1×256×256
```

### Residual block

```text
x
 ↓
Conv(32→32, 3×3)
 ↓
ReLU
 ↓
Conv(32→32, 3×3)
 ↓
+ x
```

## Loss

Start with L1:

\[
L_{L1}=\frac{1}{N}\sum_i|x_i-\hat{x}_i|
\]

L1 is intentionally simple for this phase.

## Training Configuration

Recommended starting point:

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 0
batch_size: 1
mixed_precision: optional
iterations: until convergence
```

The exact optimizer or learning rate may be changed, but every change must be logged.

## Implementation Plan

### Step 1
Select sample indices `[0]` or `[0,1]`.

### Step 2
Disable augmentation.

### Step 3
Disable random synthetic degradation.

### Step 4
Train repeatedly on exactly the same input-target pairs.

### Step 5
Log:

```text
step
loss
PSNR
SSIM
```

### Step 6
Save predictions every fixed number of iterations.

### Step 7
Inspect convergence and final visual output.

## Verification

A successful run should show:

```text
loss ↓
PSNR ↑
SSIM ↑
```

and a prediction visually approaching GT.

The exact target score is not prescribed. The acceptance criterion is that the network can strongly memorize the chosen pairs.

## Failure Diagnosis

### Loss does not decrease
Check:

- GT/NoisyLR pairing
- tensor shape
- output scaling
- optimizer
- learning rate
- gradient values

### Loss decreases but image remains wrong
Check:

- output resolution
- pixel shuffle configuration
- target scaling
- incorrect channel order

### NaN loss
Check:

- data corruption
- overflow
- mixed precision
- invalid metric code accidentally used in training

## Deliverables

```text
results/phase02_overfit/
├── loss_curve.png
├── metrics.csv
├── sample0_prediction.npy
├── sample0_visual.png
└── run_config.yaml
```

## Acceptance Gate

Do not start full training until one/two samples can be intentionally overfit.
