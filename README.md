# SEMICON AI 2026 - AI-Based Restoration of Degraded Images

This repository contains the implementation pipeline for the AI-Based Restoration of Degraded Images challenge, following a strict 13-Phase Agentic Roadmap.

## Setup

It's recommended to run scripts inside a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Phases

All scripts should be executed from the project root (`d:\semi-img-restore`).

### Phase 00: Dataset Audit
Verifies the counts, shapes, and statistical ranges of the provided `.npy` files.
```bash
venv\Scripts\python.exe src\dataset_audit.py
```

### Phase 01: Bicubic Baseline
Runs a non-neural `F.interpolate` baseline to establish the absolute PSNR, SSIM, and LPIPS floor.
```bash
venv\Scripts\python.exe src\baseline.py
```

### Phase 02: Overfit Sanity Check
Trains a tiny `ResUNet` variant on exactly 2 pairs for 2000 steps to verify gradient flow and loss convergence.
```bash
venv\Scripts\python.exe src\train_overfit.py
```

### Phase 03: Neural Baseline (ResUNet)
Trains the full `ResUNet` architecture over the 2,880 training split for 30 epochs using an `AdamW` optimizer and `CosineAnnealingLR` scheduler. Evaluates against the 320 validation split.
```bash
venv\Scripts\python.exe src\train.py
```

## Results
Results, loss curves, configurations, and visuals for each phase are saved inside the respective `results/phaseXX_.../` directories. Metrics for major neural experiments are appended to `experiments.csv`.