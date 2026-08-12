# Phase 10 — Final Model Selection & Checkpoint Freezing

## Objective
Select one final model and freeze the exact architecture, weights, preprocessing, loss configuration, and inference behavior.

## Selection Criteria

The final candidate must balance:

### Restoration quality
- PSNR ↑
- SSIM ↑
- LPIPS ↓

### Generalization
- ID validation performance
- OOD validation performance

### Efficiency
- end-to-end runtime
- peak VRAM
- parameter count

### Reliability
- deterministic inference
- no NaNs/Infs
- correct output dimensions
- correct output range

## Final Configuration Record

Create one immutable config:

```yaml
model:
  name: <selected_model>
  channels: 1
  scale: 2

loss:
  name: <selected_loss>
  weights: {...}

data:
  input_size: [128,128]
  output_size: [256,256]
  preserve_raw_noisylr: true

inference:
  clip_output: true
  clip_min: 0.0
  clip_max: 1.0
```

## Checkpoint Contents

Save:

- model weights
- architecture name/version
- config
- training seed
- library versions
- optimizer state if needed for reproducibility
- commit hash

## Final Sanity Evaluation

Run the frozen checkpoint once with no experimental code paths enabled.

Generate:

```text
final_metrics.json
final_visuals/
final_runtime.json
```

## Deliverables

```text
weights/final_model.pth
configs/final.yaml
results/phase10_final/
├── final_metrics.json
├── final_runtime.json
└── visuals/
```

## Acceptance Criterion

After this phase, the model architecture and preprocessing must not change unless a documented regression is discovered.
