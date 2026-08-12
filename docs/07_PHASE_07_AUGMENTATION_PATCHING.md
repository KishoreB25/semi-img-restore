# Phase 07 — Geometric Augmentation & Patch Training

## Objective
Improve generalization and training efficiency while preserving exact GT↔NoisyLR alignment.

## Scope

### Augmentation
- horizontal flip
- vertical flip
- 90° rotations
- 180° rotation
- 270° rotation

### Patch training
Optional aligned crops from:

```text
GT      : 2P × 2P
NoisyLR : P × P
```

For example:

```text
GT      : 128×128 patch
NoisyLR :  64×64 patch
```

## Critical Alignment Rule

The same spatial transform must be applied to both images.

Never independently crop/rotate GT and NoisyLR.

## Patch Sampler

A patch sampler should:

1. sample a valid LR crop location
2. map it to the corresponding HR coordinates
3. crop both arrays
4. apply the same geometric augmentation

## Hard Example Sampling

After a baseline exists, optionally oversample patches with:

- high gradient magnitude
- high residual error
- high local noise statistics

Do this only as a controlled experiment.

## Implementation Plan

### Step 1
Implement paired transform object.

### Step 2
Implement aligned crop sampler.

### Step 3
Verify exact geometry with synthetic coordinate patterns.

### Step 4
Compare full-image training vs patch training.

### Step 5
Test augmentation on/off.

## Verification

Create a synthetic image where each pixel encodes its coordinates. Apply an augmentation and verify that both GT and NoisyLR retain exact correspondence after scale mapping.

## Experiment Matrix

```text
A: full image, no augmentation
B: full image, augmentation
C: patch, no augmentation
D: patch, augmentation
```

## Metrics

Use the frozen validation set:

- PSNR
- SSIM
- LPIPS

Also record:

- samples/sec
- GPU memory
- convergence speed

## Deliverables

```text
src/augmentation.py
src/patch_sampler.py
configs/patch_training.yaml
results/phase07_augmentation/
```

## Acceptance Criterion

Keep patch/augmentation training only if validation generalization improves without introducing visible structural artifacts.
