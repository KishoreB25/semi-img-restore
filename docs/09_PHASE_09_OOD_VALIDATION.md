# Phase 09 — Out-of-Distribution Robustness Validation

## Objective
Estimate how well the model generalizes to image content not represented by the training subset.

KLA's hidden test includes both in-distribution and out-of-distribution content while the degradation mechanisms remain the same.

## Scope

- source/group-aware validation if metadata permits
- distribution shift analysis
- metric degradation under OOD
- failure-case clustering

## Split Strategy

### Preferred
Group/source-based holdout if the dataset exposes reliable grouping.

### Fallback
Use a deterministic stratified/random split while acknowledging that it may under-estimate OOD difficulty.

Do not invent unavailable metadata.

## Analysis

Compare:

```text
ID validation
vs
OOD validation
```

Metrics:

- PSNR
- SSIM
- LPIPS

Also compare:

- pixel intensity histograms
- gradient statistics
- texture statistics if practical

## Robustness Questions

1. Does PSNR collapse on unfamiliar structures?
2. Does SSIM remain stable?
3. Does LPIPS worsen because of texture mismatch?
4. Does the model hallucinate more on unfamiliar content?
5. Does the model overfit specific brightness distributions?

## Mitigation Experiments

Only if OOD performance is weak, test one factor at a time:

- stronger geometric augmentation
- broader synthetic degradation sampling
- patch diversity
- regularization
- lightweight architecture changes
- loss adjustments

## Deliverables

```text
results/phase09_ood/
├── id_vs_ood.csv
├── histograms/
├── visuals/
└── robustness_report.md
```

## Acceptance Criterion

The selected model must be evaluated on an OOD-like validation protocol before it is considered final.
