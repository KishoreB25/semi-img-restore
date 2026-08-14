# Reproducibility & Phase History Audit

This document traces the lineage of the final `E06-D` model and clarifies the distinction between exploratory experiments and the final, frozen model.

## 1. Phase History Summary

*   **Phase 01 (Baseline Setup):** Established the data loaders, dataset object, and basic evaluation loop. Verified the input-output dimensions (128x128 -> 256x256).
*   **Phase 02 (Overfitting Test):** Trained a simple baseline network on a 1-image dataset to verify that the training loop, loss backpropagation, and optimizer could successfully overfit and converge, proving the pipeline's basic correctness.
*   **Phase 03 (Neural Baseline):** Trained a standard CNN/UNet baseline to establish the first learned metric ceiling. 
*   **Phase 04 (Architectural Ablation - ResUNet):** Upgraded the architecture to a ResUNet. Achieved a significant performance jump to **27.82 dB** (audited). This served as the primary baseline for subsequent experiments.
*   **Phase 05 (Synthetic Data Generation):** Built an offline synthetic degradation engine (Gaussian + Speckle noise) to combat dataset limitations. Experimented with synthetic ratios.
*   **Phase 06 (Advanced Architecture - E06-D):** Designed the `AdvancedResUNet` with Squeeze-and-Excitation (SE) blocks and Dilated Bottlenecks. 
    *   *Experiments:* Ran multiple architectural ablations (E06-A, E06-B, E06-C).
    *   *Result:* The fully featured **E06-D** emerged as the undisputed winner. It was trained for 120 epochs and achieved **28.54 dB**. **This is the final model.**
*   **Phase 07 (Evaluation Audit):** Uncovered that previous phases evaluated metrics on raw network outputs (which could fall slightly outside `[0,1]`). Instituted the strict **Phase 07.1 Protocol**, mandating that predictions be clamped to `[0,1]` prior to metric calculation. All Phase 10 final results use this audited protocol.
*   **Phase 08 (Robustness Analysis):** Mined the dataset to extract standard deviation, mean, min/max, and categorical failure modes (revealing oversmoothing as the primary limitation). Generated visual grids (`FIGURES/`).
*   **Phase 09 (Texture Ablation):** Attempted to fix the Phase 08 oversmoothing failure mode by training a new E06-D candidate with an added Gradient Loss. The candidate dropped 1.06 dB globally. It was rejected.

## 2. Separation of Experiments vs. Final Model

The `E06-D` model (`results/phase06_E06-D_final/checkpoints/best_model.pth`) is the exclusive, frozen final product.

The following models/configurations were strictly **experiments** and do not represent the final pipeline:
*   The Phase 04 ResUNet.
*   Any model utilizing synthetic data (Phase 05). Synthetic data was ultimately found unnecessary/detrimental for the clean E06-D convergence.
*   Phase 09 Gradient Loss candidate.

## 3. Metric Discrepancies and Clarifications

**The Phase 06 vs Phase 07.1 Clamping Clarification:**
During Phase 06, the PSNR for E06-D was initially reported slightly higher (e.g., >28.6 dB) because the network's raw `float32` outputs were evaluated directly.
In Phase 07, an audit was performed to enforce standard image restoration rules: predictions must be clamped to the valid `[0,1]` image range before calculating PSNR. 
All results reported in Phase 10 (`FINAL_RESULTS.csv`, `ABLATION_TABLE.csv`, `FINAL_MODEL_CARD.md`) have been strictly re-evaluated using the clamped Phase 07.1 protocol to ensure mathematically rigorous and reproducible metrics.
