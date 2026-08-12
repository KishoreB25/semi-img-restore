# Phase 08 — Degradation-Specific Evaluation & Failure Analysis

## Objective
Determine whether the final candidate model actually solves each required degradation mechanism independently and jointly.

## Scope

Build controlled evaluation sets from clean validation GT:

1. Gaussian-only
2. Speckle-only
3. Downsample-only
4. Mixed degradation
5. Multiple degradation orders

## Evaluation Matrix

| Condition | Input | Expected Output |
|---|---|---|
| Gaussian | GT + Gaussian | GT-like 256×256 |
| Speckle | GT × speckle | GT-like 256×256 |
| Downsample | 2× downsample | GT-like 256×256 |
| Mixed A | Gaussian → Speckle → DS | GT-like 256×256 |
| Mixed B | DS → Speckle → Gaussian | GT-like 256×256 |

## Metrics

For each subset calculate:

- PSNR
- SSIM
- LPIPS

Also compute:

- average residual magnitude
- edge preservation score if implemented

## Visual Diagnostic Layout

For each selected sample:

```text
GT
Input
Prediction
Residual
Zoomed GT crop
Zoomed Prediction crop
Zoomed Residual crop
```

## Failure Categories

### Residual noise
The network removes insufficient noise.

### Oversmoothing
High-frequency structure disappears.

### Ringing / artificial edges
The network introduces non-existent patterns.

### Intensity bias
Prediction shifts mean intensity or histogram.

### Structural hallucination
Model invents detail not supported by GT.

## Decision Logic

If one degradation is significantly weaker:

1. inspect synthetic degradation calibration
2. inspect loss sensitivity
3. inspect architecture receptive field
4. inspect training mixture ratio
5. inspect residual maps

Do not immediately add model complexity.

## Deliverables

```text
results/phase08_degradation_eval/
├── degradation_metrics.csv
├── per_condition_visuals/
├── failure_cases/
└── analysis.md
```

## Acceptance Criterion

No major degradation mode should be silently ignored. The report must show where the model succeeds and where it fails.
