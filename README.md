<div align="center">
  <img src="assests\images\banner.png" alt="SEMI Hackathon 2026" width="600"/>
</div>

# Track 1 : Image Restoration (SEMI Hackathon 2026 Submission)

This repository contains our final submission for **Track 1 : AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## 1. Problem Statement
In semiconductor manufacturing, microscopic inspection images are used to measure and verify chip quality at every stage of production. These images must be extremely sharp and clean because a single pixel of noise or a small loss of detail can hide a defect that causes a chip to fail. 

In practice, these images are often degraded by two types of signal loss:
*   **Speckle Noise**: Random pixel-level noise that makes the image look 'grainy', pushing pixel values beyond the true image range.
*   **Spatial Resolution Reduction**: The image has been downsampled (e.g., 256x256 → 128x128), losing critical fine detail that was visible at full resolution.

The goal of this project is to develop an AI-powered restoration model capable of removing this noise and sharpening the detail back to the original resolution, enabling accurate defect detection.

## 2. Our Methodology

To solve this, we designed the **AdvancedResUNet (E06-D variant)**. It is a highly optimized, fully convolutional architecture containing exactly **1,026,766 parameters**, carefully balancing restoration quality against inference latency on edge hardware.

Key architectural innovations include:
*   **Multi-Scale Dilated Bottleneck**: Instead of a standard U-Net bottleneck, we use parallel convolutions with dilations of 1, 2, and 4. This significantly expands the model's receptive field to capture global structural context without losing spatial resolution.
*   **Squeeze-and-Excitation (SE) Blocks**: Placed at the end of every encoder and decoder block, these modules adaptively recalibrate channel-wise feature responses, allowing the network to focus heavily on high-frequency texture details rather than flat, noisy regions.
*   **Charbonnier Loss Optimization**: We trained the model using a robust Charbonnier Loss ($\epsilon=10^{-3}$) which penalized outliers less severely than standard MSE, preventing the model from aggressively blurring out critical semiconductor textures.

Our final model achieved a rigorous validation performance of **28.54 dB PSNR** and **0.7604 SSIM**.

---

## 3. Repository Structure

```
semi-img-restore/
├── README.md               # Setup and execution instructions
├── requirements.txt        # Frozen training & inference environment
├── LICENSE                 # MIT License
│
├── src/                    # Core source code (Architecture, DataLoaders, Metrics)
│   ├── resunet_advanced.py # The official E06-D Architecture
│   ├── dataset.py
│   ├── metrics.py
│   └── ...
│
├── scripts/                # Utility scripts
│   └── generate_training_curves.py 
│
├── train.py                # Reproducibility training script
├── inference.py            # STANDALONE INFERENCE SCRIPT (Competition Target)
│
├── weights/
│   └── best_model.pth      # Official frozen 1,026,766 parameter E06-D Checkpoint
│
├── outputs/
│   ├── validation/         # 320-image reproducibility output directory
│   └── test_restored/      # 400 restored outputs from the public Test_NoisyLR
│
├── Test_NoisyLR/           # Provided public test dataset (inputs)
│   └── NoisyLR/            # 400 noisy float32 .npy arrays
│
├── figures/                # Qualitative visualizations from validation
│   └── ...
│
└── docs/                   # Authoritative model cards and documentation
    ├── FINAL_MODEL_CARD.md
    ├── EVALUATION_PROTOCOL.md
    └── REPRODUCIBILITY.md
```

> [!NOTE] 
> **Where is the Training Data?** 
> The massive training dataset is intentionally excluded from this GitHub repository to comply with file size constraints. However, we have included the complete public test dataset (`Test_NoisyLR`) and our final test outputs (`outputs/test_restored/`) so you can verify our exact predictions.

## 4. Setup Instructions

Our model relies on PyTorch and standard image processing libraries. You can install our exact frozen environment using:

```bash
pip install -r requirements.txt
```

*Note:* We highly recommend running this on a CUDA-compatible environment. Our scripts will automatically fall back to CPU if CUDA is unavailable, but inference time will significantly increase.

## 5. Running Inference (Standalone Evaluator)

We have provided `inference.py` as our standalone evaluation script. It requires zero manual source code modifications and accepts raw paths directly from the command line.

**Our Important Guarantees:**
*   **Frozen Weights:** We load exactly the frozen `best_model.pth` (E06-D) checkpoint.
*   **Input Support:** We accept raw `float32` `.npy` arrays as input.
*   **Safe Clamping:** We produce strictly clamped `[0,1]` float32 predictions internally to prevent overflow.
*   **Blind Inference:** We do not compute Ground Truth metrics during the inference pass.
*   **Filename Preservation:** We automatically write the output file with the exact same base name as the input file (e.g., `000001.npy` -> `000001.npy`), perfectly satisfying your filename preservation requirement.

### Execution Command (H100 Benchmark / Private Test Set)

To evaluate our model on your hidden test dataset, simply clone our repository and execute the script by providing your custom `--input_dir` and `--output_dir` paths:

```bash
python inference.py --input_dir /path/to/hidden_test_images --output_dir /path/to/restored
```

### Reproducing our Public Test Results

We ran inference on the public `Test_NoisyLR` dataset using the exact command below. The resulting 400 restored float32 `.npy` images are included in this repository under `outputs/test_restored/`:

```bash
python inference.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/test_restored
```

### Configurable Output Format
By default, we output strictly unquantized native `float32` `.npy` arrays to preserve maximum detail. If your evaluation framework ultimately requires PNG or TIFF image files, you can simply append the `--output_format` flag:

```bash
python inference.py --input_dir /path/to/test_images --output_dir /path/to/restored --output_format png
```
*Supported formats: `npy` (default), `png`, `tif`.*

## 6. Training Reproducibility

We have already provided our final model weights in `weights/best_model.pth`, so training from scratch is **not required** for inference. However, if you wish to verify our reproducibility, you can run our training script directly from the repository root:

```bash
python train.py
```

This will automatically load our canonical Phase 10.1 Ground Truth hyperparameters (`AdamW`, `CosineAnnealingLR`, `lr=2e-4`, Charbonnier Loss `eps=1e-3`, batch size 16) and train our E06-D architecture for 120 epochs.
