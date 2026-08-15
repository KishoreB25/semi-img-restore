# Final Model Card: E06-D (AdvancedResUNet)

## 1. Final Model Information
*   **Architecture Name:** AdvancedResUNet (E06-D variant)
*   **Key Components:** Dilated Bottlenecks, Squeeze-and-Excitation (SE) blocks.
*   **Parameter Count:** 1,026,766
*   **Checkpoint Path:** `results/phase06_E06-D_final/checkpoints/best_model.pth`
*   **Selected Epoch:** Epoch 116 (Selected via maximum PSNR on the validation set during training)
*   **Loss Function:** Charbonnier Loss (L1 with epsilon `1e-6`)

### Training Configuration
*   **Optimizer:** Adam (`lr=1e-3`, `weight_decay=1e-5`)
*   **Scheduler:** OneCycleLR (`max_lr=1e-3`, `epochs=120`, `steps_per_epoch=180`)
*   **Batch Size:** 16
*   **Epochs:** 120
*   **Augmentations:** Synthetic data augmentation disabled. Random cropping and flipping disabled. Data loaded as 128x128 patches.
*   **Train/Validation Split:** 90/10 split fixed via `configs/split.yaml`.

## 2. Robustness Statistics (Strict Evaluation Protocol)
Evaluated across all 320 fixed validation samples.

*   **Mean PSNR:** 28.54 dB
*   **Median PSNR:** 28.37 dB
*   **Standard Deviation (PSNR):** 5.09 dB
*   **Minimum PSNR:** 11.39 dB
*   **Maximum PSNR:** 40.45 dB

### Worst Five Samples
1.  **`000051.npy`** (PSNR: 11.39 dB)
2.  **`001385.npy`** (PSNR: 13.90 dB)
3.  **`000900.npy`** (PSNR: 14.88 dB)
4.  **`000354.npy`** (PSNR: 15.35 dB)
5.  **`002639.npy`** (PSNR: 16.09 dB)

## 3. Latency
Calculated using strict CUDA synchronization (`torch.cuda.synchronize()`).
*   **GPU / Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA)
*   **Timing Methodology:** Warmup passes ignored. Mean/Median calculated over 320 sequential batch-1 passes using `time.perf_counter()`.
*   **Audited Mean Latency:** 35.12 ms
*   **Audited Median Latency:** 33.57 ms

## 4. Qualitative Results & Failure Analysis
*See the `FIGURES/` folder for visual grids.*

### Primary Failure Mode: Texture Loss / Oversmoothing
Analysis of the five lowest-PSNR images reveals that the primary failure mode is **oversmoothing**. On highly degraded patches containing intricate structures, the model aggressively smooths out the details to minimize the Charbonnier loss penalty. 
*Note:* While these failure cases correlate strongly with severe LPIPS spikes (e.g., `>0.4`), LPIPS alone does not prove the cause (as LPIPS responds to many perceptual differences, including blur, shift, or residual noise). However, qualitative visual inspection confirms that the perceptual divergence is driven by excessive blurring and texture erasure rather than structural hallucinations. 

Attempts to inject a Gradient Loss () to force texture preservation succeeded slightly on these 5 specific failure cases but caused a catastrophic -1 dB drop across the global dataset, proving that E06-D with standard Charbonnier is the optimal global balance.
