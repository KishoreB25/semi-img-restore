# Phase 11 — Inference Pipeline, Runtime Optimization & Clean-Environment Validation

## Objective
Build the exact standalone inference pipeline required by KLA and optimize **end-to-end** throughput.

KLA defines runtime as including disk I/O, preprocessing, CPU↔GPU transfers, model execution, post-processing and saving.

## Required Interface

The script must accept:

```bash
python inference.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output
```

No source-code edits or notebook modifications should be required.

## Inference Flow

```text
input directory
    ↓
find every .npy
    ↓
load float32
    ↓
add channel dimension
    ↓
CPU tensor
    ↓
GPU transfer
    ↓
model inference
    ↓
GPU → CPU
    ↓
remove channel dimension
    ↓
clip [0,1]
    ↓
save .npy
```

## Runtime Optimizations

Test systematically:

### 1. `torch.inference_mode()`
No gradients.

### 2. Automatic mixed precision
Use FP16/AMP where numerically safe.

### 3. Batch inference
Benchmark batch sizes such as:

```text
1, 2, 4, 8, 16
```

subject to memory.

### 4. Pinned memory / non-blocking transfers
Use only when the DataLoader/path actually benefits.

### 5. Efficient model loading
Load weights exactly once.

### 6. Optional compilation
Evaluate `torch.compile` only after correctness is frozen.

## Timing Protocol

Measure at least:

```text
I/O load time
preprocess time
CPU→GPU time
auto/model time
GPU→CPU time
postprocess time
save time
total time
```

Report:

- total runtime for N images
- average/image
- throughput images/sec
- batch size
- GPU
- software versions

## Output Contract

Each input `.npy` must produce the required corresponding output file using the official naming/format convention.

Final output properties:

```text
shape = (256,256)
dtype = float32
range = [0,1]
```

## Clean-Environment Test

Create a clean environment from `requirements.txt`.

Run:

```bash
python inference.py --input-dir test_sample --output-dir clean_run
```

No manual path edits.

## Failure Tests

- missing input directory
- empty input directory
- malformed `.npy`
- unexpected shape
- NaN/Inf input
- missing model weights
- insufficient VRAM
- output directory absent

The script should fail with useful, explicit messages.

## Deliverables

```text
inference.py
requirements.txt
results/phase11_runtime/runtime_report.md
results/phase11_runtime/timings.csv
results/phase11_runtime/clean_run/
```

## Acceptance Criterion

The final inference script must run from a clean environment using only input/output arguments and produce the correct output set without manual intervention.
