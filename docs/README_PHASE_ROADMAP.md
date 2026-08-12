# SEMICON AI 2026 — Agentic Implementation Roadmap

## Purpose

This directory contains phase-specific implementation specifications for an agentic IDE workflow (for example, Antigravity). Each phase is deliberately written as an execution contract: objective, scope, folder structure, implementation steps, verification, testing, deliverables and acceptance criteria.

## Phase Order

| Phase | File | Goal | Status |
|---|---|---|---|
| 00 | `00_PHASE_00_DATASET_AUDIT.md` | Dataset reconnaissance | Completed |
| 01 | `01_PHASE_01_DATA_PIPELINE_BASELINE.md` | Loader + bicubic baseline | Next |
| 02 | `02_PHASE_02_OVERFIT_SANITY.md` | Pipeline sanity | Next |
| 03 | `03_PHASE_03_TRAIN_VAL_SPLIT_BASELINE.md` | First neural baseline | Next |
| 04 | `04_PHASE_04_LOSS_EXPERIMENTS.md` | Loss search | Planned |
| 05 | `05_PHASE_05_SYNTHETIC_DEGRADATION.md` | Synthetic robustness | Planned |
| 06 | `06_PHASE_06_ARCHITECTURE_EXPERIMENTS.md` | Architecture search | Planned |
| 07 | `07_PHASE_07_AUGMENTATION_PATCHING.md` | Augmentation/patch efficiency | Planned |
| 08 | `08_PHASE_08_DEGRADATION_SPECIFIC_EVALUATION.md` | Failure analysis | Planned |
| 09 | `09_PHASE_09_OOD_VALIDATION.md` | OOD robustness | Planned |
| 10 | `10_PHASE_10_FINAL_MODEL_SELECTION.md` | Freeze final checkpoint | Planned |
| 11 | `11_PHASE_11_RUNTIME_INFERENCE_HARDENING.md` | Production inference | Planned |
| 12 | `12_PHASE_12_SUBMISSION_PACKAGING.md` | Final submission | Planned |

## Agentic Execution Rules

The agent should follow these rules in every phase:

1. Read the current phase file completely before modifying code.
2. Inspect the existing repository before creating new files.
3. Do not silently change requirements from previous phases.
4. Keep all experiment configurations explicit.
5. Never overwrite the frozen validation split.
6. Never use hidden/test inputs for training or model selection.
7. Preserve raw NoisyLR values; do not silently clip or per-image normalize inputs.
8. Clip final restored outputs to `[0,1]` before saving.
9. Run verification tests after every material implementation change.
10. Update the experiment log after every completed experiment.
11. Prefer one-variable-at-a-time comparisons.
12. Keep training and inference code runnable from the command line.
13. Do not optimize runtime before correctness is established.
14. Do not increase model complexity merely because a metric moved slightly; compare the full quality/runtime tradeoff.

## Core Scientific Contract

### Input

```text
NoisyLR
shape = 128×128×1
float32
raw range preserved
```

### Output

```text
Restored GT-like image
shape = 256×256×1
float32
clipped to [0,1] before save
```

### Required benchmark degradations

- Speckle noise
- Additive Gaussian noise
- Downsampling

Order may vary. The model should restore them jointly or through a justified staged design.

## Experiment Registry

Maintain one CSV/JSON registry:

```text
experiment_id
phase
git_commit
model
loss
augmentation
synthetic_ratio
seed
batch_size
learning_rate
epochs_or_steps
psnr
ssim
lpips
runtime_ms_per_image
peak_vram_mb
notes
```

## Final Scientific Question

> Which lightweight restoration architecture and loss combination best reconstructs the clean 256×256 image from the degraded 128×128 observation while preserving real semiconductor structure, generalizing to unfamiliar content, and maintaining efficient end-to-end inference?

## Official Source Basis

The specifications are grounded primarily in the uploaded KLA Problem Statement and KLA Image Restoration Task / webinar material. The source documents require paired restoration, the three official degradations, validation without leakage, PSNR/SSIM/LPIPS reporting, runtime measurement, reproducible training/inference code, model weights/configuration and a clean submission package.
