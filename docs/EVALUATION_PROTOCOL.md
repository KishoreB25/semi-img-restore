# Phase 07.1: Official Evaluation Protocol

To ensure strict comparability across all current and future experiments, this document defines the exact evaluation protocol to be used.

## 1. Data Pipeline
*   **Validation Set:** The fixed 10% validation split (320 images/batches) from the real dataset.
*   **Input Data:** Raw `float32` NoisyLR arrays directly from `.npy` files.
*   **No Normalization:** NO per-image normalization, min-max scaling, or mean subtraction is applied prior to model inference.

## 2. Model Inference & Preprocessing
*   **Batch Size:** 1 (for both metrics and latency measurements).
*   **Upsampling:**
    *   *Bicubic Baseline:* `torch.nn.functional.interpolate(..., size=(256, 256), mode="bicubic", align_corners=False)`
    *   *Neural Models:* Output directly shaped `(1, 1, 256, 256)`.
*   **Clipping (CRITICAL):** All model predictions are strictly clamped using `torch.clamp(pred, 0.0, 1.0)` *before* any metrics are calculated. The Ground Truth (GT) is identically clamped to `[0.0, 1.0]`.

## 3. Metric Calculations
*   **PSNR:** Calculated using PyTorch's Mean Squared Error (`F.mse_loss`) on the clipped `[0,1]` tensors, assuming a maximum pixel value of `1.0`. Formula: `10 * log10(1.0 / MSE)`.
*   **SSIM:** Calculated using `skimage.metrics.structural_similarity` on detached CPU numpy arrays with `data_range=1.0`.
*   **LPIPS:**
    *   *Package:* `lpips`
    *   *Backbone:* AlexNet (`net='alex'`)
    *   *Preprocessing:* Grayscale inputs are replicated to 3 channels (`.repeat(1, 3, 1, 1)`).
    *   *Scaling:* Tensors are scaled from `[0, 1]` to `[-1, 1]` before being passed into the LPIPS backbone.

## 4. Latency Audit Protocol
Because CUDA operations are asynchronous, latency must be measured with explicit synchronization barriers:
1.  Set model to `.eval()` mode.
2.  Perform 10 warmup iterations within a `torch.no_grad()` context.
3.  For $N=100$ iterations:
    *   `torch.cuda.synchronize()`
    *   Start timer.
    *   Execute model forward pass.
    *   `torch.cuda.synchronize()`
    *   Stop timer.
4.  Record each iteration's time. Report the **Mean**, **Median**, and **Standard Deviation**.
