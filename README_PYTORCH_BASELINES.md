# PyTorch compact MEG baselines

This branch adds strict no-calibration LOSO baselines inspired by the papers we discussed:

- `eegnet`: EEGNet-style temporal + depthwise spatial CNN.
- `lfcnn`: LF-CNN-style spatial latent filters plus temporal filtering.
- `varcnn`: VAR-CNN-style lag/dynamics CNN.
- `hgrn`: HGRN-style spatial projection + GRU + attention pooling.

The runner expects a trial tensor `X` shaped `trials x channels x time`, labels `y`, and subject IDs `subjects`. For each held-out subject, the code fits normalization, trains the model, and does early stopping using source subjects only. The target subject is used only for final evaluation.

## Export from the existing MATLAB pipeline

```matlab
exportTrialTensorForPyTorch('data', [1:4, 6, 8:10, 13:27], ...
    'meg_trials_for_pytorch.mat', 0.2, 0.2, nan, inf, [0, inf]);
```

The exporter reuses `preprocessFeatures.m`, so filtering, downsampling, and window extraction stay aligned with the existing MATLAB baselines. Null-window pseudo-trials are omitted because these neural baselines are multiclass stimulus classifiers.

## Run the neural LOSO baselines

```bash
python pytorch_meg_baselines.py \
  --data meg_trials_for_pytorch.mat \
  --models eegnet lfcnn varcnn hgrn \
  --epochs 80 \
  --patience 15 \
  --batch-size 64 \
  --output-prefix results/pytorch_loso
```

The output is written to `results/pytorch_loso.csv` and `results/pytorch_loso.json` with fold-level accuracy, balanced accuracy, chance level, held-out subject, and early-stopping epoch.

## Smoke test

```bash
python pytorch_meg_baselines.py --smoke-test
```

I ran this smoke test locally with synthetic data. It verifies that all four model families train and evaluate end-to-end, but it is not a real MEG benchmark.
