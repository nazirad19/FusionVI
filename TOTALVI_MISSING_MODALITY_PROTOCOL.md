# Frozen protocol: missing-modality training for totalVI

Status: frozen before the first development-split sweep

Branch: `totalvi-missing-modality-followup`

Primary question: Can supervised whole-panel dropout improve RNA-only protein recovery without reducing within-cell-type biological ranking?

This is a follow-up project. It does not alter the completed FusionVI course report or its conclusions.

## Data boundary

Model and hyperparameter selection use only the existing `dev_random` and `dev_tissue` benchmark objects derived from SLN111-D1. The real SLN111-D2 target mouse is absent from both objects. Development target cells retain RNA for unsupervised VAE training, while their full protein panel is zeroed in the training matrix and preserved only in `protein_truth` for evaluation. This matches the final complete-missing-panel setting.

`dev_random` is an optimistic within-mouse reference. `dev_tissue` is a harder within-mouse tissue shift. Neither is described as the cross-mouse performance ceiling.

The `paper` benchmark is not inspected for model selection. It is used once, after the method and hyperparameters are frozen.

## Prespecified development arms

All arms use the published totalVI joint encoder width of 256, the same 20-dimensional latent space, decoder, likelihoods, optimizer and 500-epoch budget.

| Arm | Whole-panel dropout probability |
|---|---:|
| `totalvi` | 0.00 |
| `totalvi_moddrop_p25` | 0.25 |
| `totalvi_moddrop_p50` | 0.50 |
| `totalvi_moddrop_p75` | 0.75 |

Whole-panel dropout changes only the encoder input for a sampled source cell. Its original measured protein counts remain targets of the totalVI protein likelihood.

Development seeds are `2026`, `2027`, `2028` and `2029`. Four seeds estimate direction and stability; development decisions do not use p-values.

## Readouts

The primary metric is mean per-protein RMSLE from `likelihood_mean`, the likelihood-consistent expected protein count computed from `py_norm_` and transformed with `log1p`.

The protected secondary metric is mean within-cell-type, per-protein Spearman correlation. Overall Spearman, residual error, bias and marker AUROC are descriptive.

Every development checkpoint is decoded in the source batch, using the learned SLN111-D1 protein efficiency, as in the paper benchmark. Before selection, record the median D1 efficiency for every arm and seed. A median outside `[0.25, 0.50]`, or an absolute difference greater than `0.10` from its paired totalVI seed, triggers a training audit before any result is interpreted.

## Exact masking selection rule

For each dropout arm and each development split, calculate the paired seed difference

`delta RMSLE = dropout-arm RMSLE - totalVI RMSLE`

and

`delta within-CT Spearman = dropout-arm Spearman - totalVI Spearman`.

A dropout rate **passes** only when both conditions hold on both `dev_random` and `dev_tissue`:

1. mean paired `delta RMSLE <= -0.0010`;
2. mean paired `delta within-CT Spearman >= -0.0020`.

If multiple rates pass, select the rate with the lowest average paired RMSLE difference across the two splits. If two rates are within `0.0005`, select the lower dropout probability.

If no rate passes, masking is recorded as a negative result and no confirmatory masking run is performed on `paper`.

A rate is **neutral** when, on both splits, its mean paired RMSLE difference lies in `[-0.0010, +0.0010]` and its within-cell-type Spearman difference is at least `-0.0020`. If no rate passes but at least one is neutral, one stop-gradient distillation arm may be evaluated on the development splits. That arm is explicitly exploratory and is not treated as a prespecified confirmatory result. If every rate is harmful on either split, development stops.

## Conditional distillation experiment

Distillation is implemented only after the masking gate above is evaluated. A frozen, stop-gradient RNA-plus-protein teacher supplies its posterior parameters and likelihood-consistent protein mean. The student receives the masked RNA-only route. Any latent KL and protein-prediction loss weights are chosen using development data only and recorded before the final benchmark.

The novelty claim is restricted to applying posterior distillation to complete-missing-panel transfer. Posterior alignment itself is not claimed as new.

## Conditional residual experiment

A residual head is attempted only after the base dropout/distillation model is frozen. It predicts cell-specific residuals rather than a global affine scale. Training residuals must come from outer-fold predictions for which that cell's protein panel was unavailable to the corresponding VAE fit. Three outer folds are used to control computational cost. The residual head is not fitted to in-sample D1 protein predictions.

## Final benchmark and inference

The final primary contrast is the selected single-checkpoint method versus published single-checkpoint totalVI on `paper`, using `likelihood_mean` RMSLE and 16 paired seeds (`2026` through `2041`). The selected method is successful only if:

1. its mean paired RMSLE difference is below zero;
2. the 95% paired confidence interval is entirely below zero;
3. within-cell-type Spearman is not reduced by more than `0.0020` on average.

Holm correction is applied across all additional arm comparisons. The primary contrast is identified in advance and reported separately. Seed ensembles are optional reference rows only; any ensemble comparison uses the same number of checkpoints for totalVI and the candidate method.

All arms and development outcomes are reported, including negative results. No dropout rate, loss weight or stopping rule is changed after inspecting the `paper` target results.

## Frozen commands

```powershell
.\run_controls.ps1 -Benchmark dev_random -Arms totalvi,totalvi_moddrop_p25,totalvi_moddrop_p50,totalvi_moddrop_p75 -Seeds 2026,2027,2028,2029
.\run_controls.ps1 -Benchmark dev_tissue -Arms totalvi,totalvi_moddrop_p25,totalvi_moddrop_p50,totalvi_moddrop_p75 -Seeds 2026,2027,2028,2029

python src\evaluate_calibration.py --benchmark dev_random --arms totalvi totalvi_moddrop_p25 totalvi_moddrop_p50 totalvi_moddrop_p75 --seeds 2026 2027 2028 2029 --tag moddrop_dev_v1
python src\evaluate_calibration.py --benchmark dev_tissue --arms totalvi totalvi_moddrop_p25 totalvi_moddrop_p50 totalvi_moddrop_p75 --seeds 2026 2027 2028 2029 --tag moddrop_dev_v1
```

The tagged outputs are preserved alongside the untagged convenience outputs produced by `run_controls.ps1`.
