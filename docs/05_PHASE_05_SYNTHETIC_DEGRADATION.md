# Phase 05 — Synthetic Degradation Engine & Augmented Training Data

## Objective
Increase robustness by generating additional degraded pairs from clean GT images using the official benchmark degradation mechanisms.

KLA explicitly allows additional synthetic degraded pairs created from provided GT images.

## Scope

### In scope
- synthetic Gaussian noise
- synthetic speckle noise
- 2× downsampling
- random order of degradations
- parameter sampling based on observed/training ranges
- synthetic/real data mixing

### Out of scope
- unrelated artifacts not in the benchmark
- blind addition of blur/JPEG/rain/etc. unless separately justified as a future robustness study

## Degradation Operators

### Gaussian noise

\[
y = x + n,\qquad n\sim\mathcal{N}(0,\sigma^2)
\]

Parameterize `sigma` as a configurable distribution.

### Speckle noise

A practical benchmark-oriented form is multiplicative:

\[
y = x\odot n
\]

where `n` is sampled from the chosen speckle distribution.

The exact distribution/parameterization must be documented from the official dataset analysis rather than invented without calibration.

### Downsampling

For the verified dataset:

```text
256×256 → 128×128
```

Use the same or a justified approximation of the benchmark's downsampling behavior.

## Randomized Order

Generate examples such as:

```text
GT → Gaussian → Speckle → Downsample
GT → Speckle → Gaussian → Downsample
GT → Downsample → Gaussian → Speckle
GT → Gaussian → Downsample → Speckle
GT → Speckle → Downsample → Gaussian
GT → Downsample → Speckle → Gaussian
```

Do not train a degradation-order classifier.

## Parameter Calibration

Use the official train distribution to estimate plausible ranges.

For every synthetic sample, log optionally:

```text
sigma_gaussian
speckle_parameter
order
seed
```

## Data Mixing Strategy

Start with:

```text
Real : Synthetic = 1 : 1
```

Then evaluate alternatives such as:

```text
1:0
1:1
1:2
1:3
```

The ratio must be treated as an experiment.

## Folder Structure

```text
src/
└── degradation.py

configs/
└── synthetic_degradation.yaml

results/phase05_synthetic/
├── generated_samples/
├── statistics.csv
└── comparison.csv
```

## Verification

### Operator-level tests

1. Gaussian-only output changes statistics but preserves general structure.
2. Speckle-only output shows multiplicative intensity-dependent variation.
3. Downsample output is exactly `128×128`.
4. Mixed operators preserve expected final dimensions.
5. Random order is reproducible under a fixed seed.

### Distribution tests

Compare synthetic and real NoisyLR:

- min/max
- histogram
- mean/std
- selected quantiles

The synthetic distribution should be plausible, not necessarily identical.

## Training Experiment

Compare:

```text
real only
vs
real + synthetic
```

on the same validation set.

## Deliverables

```text
src/degradation.py
configs/synthetic_degradation.yaml
results/phase05_synthetic/statistics.csv
results/phase05_synthetic/comparison.csv
results/phase05_synthetic/sample_visuals/
```

## Acceptance Criterion

Synthetic augmentation is retained only if it improves or meaningfully stabilizes validation/OOD performance without unacceptable runtime or training instability.
