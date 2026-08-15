# Image Restoration (KLA Competition Submission)

This repository contains the final Phase 12 codebase for the semi-supervised image restoration competition. The final architecture is **AdvancedResUNet (E06-D variant)**, characterized by multi-scale Dilated Bottlenecks and Squeeze-and-Excitation (SE) blocks.

## 1. Repository Structure

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
│   └── validation/         # 320-image reproducibility output directory
│
├── figures/                # Qualitative visualizations from validation
│   └── ...
│
└── docs/                   # Authoritative model cards and documentation
    ├── FINAL_MODEL_CARD.md
    ├── EVALUATION_PROTOCOL.md
    └── REPRODUCIBILITY.md
```

## 2. Setup Instructions

The repository relies on PyTorch and standard image processing libraries. Install the exact frozen environment using:

```bash
pip install -r requirements.txt
```

*Note:* Ensure you have a CUDA-compatible environment. The scripts will automatically fallback to CPU if CUDA is unavailable, but inference time will significantly increase.

## 3. Running Inference (Standalone Evaluator)

The competition evaluation script is `inference.py`. It requires zero manual source code modifications and accepts raw paths.

**Important Guarantees:**
*   Loads exactly the frozen `best_model.pth` E06-D.
*   Takes raw `float32` `.npy` arrays as input.
*   Produces strictly clamped `[0,1]` float32 predictions internally.
*   Does not compute Ground Truth metrics during the inference pass.
*   **Filename Preservation:** The script automatically writes the output file with the exact same base name as the input file (e.g., `000001.npy` -> `000001.npy`), satisfying the filename preservation requirement.

### Execution Command (H100 Benchmark)

The KLA Judges will clone the repository on an NVIDIA H100 GPU and execute the script passing their hidden test dataset directories:

```bash
python inference.py --input_dir /path/to/hidden_test_images --output_dir /path/to/restored
```

### Configurable Output Format
By default, the script outputs strictly unquantized native `float32` `.npy` arrays. If the competition framework ultimately demands PNG or TIFF image files, append the `--output_format` flag:

```bash
python inference.py --input_dir /path/to/test_images --output_dir /path/to/restored --output_format png
```
*Supported formats: `npy` (default), `png`, `tif`.*

## 4. Training Reproducibility

The final model weights are already provided in `weights/best_model.pth`. Training from scratch is **not required** for inference. However, if reproducibility verification is requested, the training script can be run directly from the repository root:

```bash
python train.py
```

This will automatically load the canonical Phase 10.1 Ground Truth hyperparameters (`AdamW`, `CosineAnnealingLR`, `lr=2e-4`, Charbonnier Loss `eps=1e-3`, batch size 16) and begin a 120-epoch training cycle.