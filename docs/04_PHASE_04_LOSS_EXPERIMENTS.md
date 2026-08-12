# Phase 04 — Loss Function Experiments

## Objective
Determine which training objective best balances pixel fidelity, structure preservation and fine detail without introducing artificial patterns.

KLA requires reporting PSNR, SSIM and LPIPS. The final benchmark uses a hidden weighted combination, so optimization must not focus on one metric alone.

## Scope

Test loss functions in isolation while keeping:

- dataset split
- architecture
- augmentation
- optimizer
- training budget

fixed as much as possible.

## Experiment Ladder

### E04-A — L1

\[
L_{L1}=\frac{1}{N}\sum_i|x_i-\hat{x}_i|
\]

### E04-B — Charbonnier

\[
L_{char}=\frac{1}{N}\sum_i\sqrt{(x_i-\hat{x}_i)^2+\epsilon^2}
\]

Use a small `epsilon`, e.g. `1e-3`, and keep it configurable.

### E04-C — Charbonnier + SSIM

\[
L=\lambda_cL_{char}+\lambda_s(1-SSIM)
\]

Example starting weights:

```text
lambda_c = 0.8
lambda_s = 0.2
```

Treat these as initial values, not fixed truth.

### E04-D — Add gradient loss

Compute image gradients with Sobel or finite-difference operators:

\[
L_{grad}=|\nabla_x\hat{x}-\nabla_xx|_1 + |\nabla_y\hat{x}-\nabla_yx|_1
\]

Combined:

\[
L=\lambda_cL_{char}+\lambda_s(1-SSIM)+\lambda_gL_{grad}
\]

### E04-E — Optional frequency loss

\[
L_{freq}=\left\||\mathcal{F}(\hat{x})|-|\mathcal{F}(x)|\right\|_1
\]

Test this only after the spatial-domain losses are stable.

## Model/Layers

Use the Phase 03 model unchanged.

Do not change architecture while comparing losses.

## Verification

For every loss experiment, record:

```text
loss definition
weights
optimizer
learning rate
number of epochs/steps
best PSNR
best SSIM
best LPIPS
runtime
visual observations
```

## Failure Analysis

### Higher PSNR but visibly blurry
Potentially over-optimized for pixel error.

### Better LPIPS but artificial texture
Potential perceptual hallucination; not acceptable for inspection imagery without evidence that structure is correct.

### Better SSIM but noisy residual
Loss may preserve structure while under-penalizing residual noise.

## Deliverables

```text
results/phase04_losses/
├── experiment_matrix.csv
├── visual_comparison.png
├── best_loss_config.yaml
└── checkpoints/
```

## Acceptance Criterion

Select the loss based on the joint metric profile and visual inspection, not on one metric alone.
