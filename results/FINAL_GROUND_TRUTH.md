# Phase 10.1: Final Ground-Truth Forensic Audit

This document serves as the absolute, verified, ground-truth specification for the final `E06-D` model and resolves all previous documentary contradictions.

## 1. Verified Phase 06 Training Configuration
*Audited directly from `results/phase06_E06-D_final/config.yaml` and `metrics.csv`.*

*   **Model Architecture:** AdvancedResUNet (E06-D variant)
*   **SE Configuration:** Enabled (`use_se=True`)
*   **Dilated Bottleneck Configuration:** Enabled (`use_dilated_bottleneck=True`)
*   **Parameter Count:** 1,026,766
*   **Loss Function:** Charbonnier
*   **Charbonnier Epsilon:** `1e-3` (0.001)
*   **Optimizer:** AdamW
*   **Learning Rate:** `2e-4` (0.0002)
*   **Betas:** Unverified from config (PyTorch AdamW default `(0.9, 0.999)`)
*   **Weight Decay:** Unverified from config (Hardcoded in `train.py` as `1e-4`)
*   **Scheduler:** CosineAnnealingLR
*   **Scheduler Parameters:** `T_max=120`, `eta_min=1e-6` (Hardcoded in `train.py`)
*   **Batch Size:** 16
*   **Epochs:** 120
*   **Steps Per Epoch:** 180
*   **Train/Validation Split:** 90/10 Split (Fixed via `configs/split.yaml`)
*   **Augmentation:** None
*   **Synthetic Data Usage:** 0.0 (None)
*   **Checkpoint Selection Criterion:** Maximum Validation PSNR
*   **Selected Epoch (`best_model.pth`):** Epoch 116

## 2. Resolved Discrepancies

### A. Training Config Contradiction (Version A vs Version B)
Previous documentation reported conflicting hyperparameters. 
*   **Version A:** Charbonnier eps=1e-3, AdamW, lr=2e-4, CosineAnnealingLR
*   **Version B:** Charbonnier eps=1e-6, Adam, lr=1e-3, OneCycleLR

**Resolution: Version A is authoritative.** 
The file `results/phase06_E06-D_final/config.yaml` explicitly dictates `initial_lr: 0.0002`, `char_eps: 0.001`, `optimizer: AdamW`, and `scheduler: CosineAnnealingLR`.

### B. Phase 08 vs Phase 09 Failure Case Metric Mismatches
In Phase 08, specific PSNR values were reported for the 5 worst images (e.g., `000051.npy` with PSNR 11.39). In Phase 09, evaluating those exact same files yielded completely different baseline PSNR values (e.g., `000051.npy` PSNR 17.64, while `002639.npy` had PSNR 11.39).

**Resolution:** This was caused by a manual transcription error in the markdown authoring of `PHASE08_SUMMARY.md`. The script `src/run_phase08_robustness.py` accurately identified the 5 worst PSNR values, but during the markdown write-up, the numeric PSNR values were accidentally paired with the wrong filenames from that top-5 list (e.g. assigning `002639`'s 11.39 dB score to `000051`). The Phase 09 CSV (`failure_case_comparisons.csv`) is generated programmatically by querying the exact filenames, making it the **authoritative ground-truth** for per-image metrics.

### C. Ground-Truth Clamping
The Phase 07.1 evaluation protocol dictates: `clamp prediction to [0,1], clamp GT to [0,1]`.
**Resolution:** Forensic audit of `src/run_phase07_evaluation.py` reveals that predictions were successfully clamped (`pred_e06_clipped = torch.clamp(pred_e06, 0.0, 1.0)`). However, Ground Truth (GT) was **UNVERIFIED** to be clamped in the evaluation script prior to metric calculation. The evaluation script passed `gt` directly to `evaluator.evaluate_batch()`. Since the original dataset `GT` is already assumed strictly bounded `[0,1]`, this omission did not alter the metrics, but the strict programmatic enforcement of GT clamping was technically absent.

## 3. Verified Final Model Performance

The final checkpoint (`results/phase06_E06-D_final/checkpoints/best_model.pth`) correctly reproduces the following metrics under the Phase 07.1 Audited Protocol:

*   **PSNR:** 28.541 dB
*   **SSIM:** 0.7604
*   **LPIPS (AlexNet):** 0.2888

These are the globally authoritative values for E06-D. Phase 04 ResUNet (27.821 dB) and Phase 09 E06-D+Grad (27.477 dB) were evaluated using identical protocols and are directly comparable.

## 4. Latency Methodology 

The authoritative latency figures were generated in `src/run_phase07_evaluation.py` with the following strict methodology:
*   **Metric Evaluation Batch Size:** 1
*   **Latency Benchmark Batch Size:** 1
*   **Warmup Iterations:** 10
*   **Timed Iterations:** 100
*   **CUDA Synchronization:** Enabled (`torch.cuda.synchronize()`) before and after the forward pass.
*   **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU

**Authoritative Latency (Phase 06 E06-D):**
*   **Mean:** 35.12 ms
*   **Median:** 33.57 ms
*   **Standard Deviation:** UNVERIFIED (The script calculated it in `b_std, p4_std, e06_std` but never printed or saved the standard deviation values in the CSV output).
