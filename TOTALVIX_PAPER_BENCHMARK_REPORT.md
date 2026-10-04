# Audit of the totalVI protein-imputation readout

## Research question

Does a D1-trained affine head improve totalVI on the original SLN111 missing-protein benchmark after totalVI's learned protein-efficiency parameter is included in the prediction?

## Why the audit was necessary

The first updated-library reproduction scored totalVI using rates returned by scvi-tools 1.4.2 `get_normalized_expression()`. This produced mean protein RMSLE 1.0601, much worse than the totalVI authors' published Seurat prediction at 0.6054 and inconsistent with the paper's conclusion.

Inspection of scvi-tools 1.4.2 showed that totalVI learns `log_per_batch_efficiency`. The reconstruction likelihood multiplies foreground and background protein rates by `exp(log_per_batch_efficiency)` and exposes those rates as `py_norm_`. `get_normalized_expression()` instead reads the unscaled `py_` rates. That helper returns a normalized representation, but it is not the likelihood-consistent expected count needed when scoring against observed protein counts.

Across the 16 saved models, the mean source-batch efficiency was 0.3679 (range across all seed-protein values 0.3649–0.3775). Omitting it inflated predicted protein counts by approximately 2.72-fold.

## Benchmark design

The experiment follows the totalVI paper's Figure 3 missing-protein task. SLN111-D1 supplies RNA and 110 measured proteins. Every SLN111-D2 protein is hidden during training and retained only for final scoring. Models use 4,005 genes, 110 proteins, a 20-dimensional latent space, learning rate 0.004, batch size 256, a 500-epoch maximum and 25 posterior samples decoded as D1. Sixteen saved initializations were evaluated; the paper used 30.

Four totalVI readouts were compared:

1. **Helper output, efficiency omitted:** `log1p(E[y])` computed from `py_`.
2. **Likelihood-consistent totalVI:** the same expectation computed from `py_norm_`, including the model-owned efficiency.
3. **D1 affine head:** fitted to the unscaled helper output using D1 proteins only.
4. **Likelihood-consistent totalVI + D1 head:** the same affine head applied after efficiency restoration.

No D2 protein value was used to fit a model or head.

## Results

| Readout | Mean protein RMSLE | Absolute bias | Residual SD | Spearman |
|---|---:|---:|---:|---:|
| Helper output, efficiency omitted | 1.0601 | 0.9056 | 0.5411 | 0.3262 |
| **Likelihood-consistent totalVI** | **0.5650** | 0.1358 | 0.5417 | 0.3262 |
| D1 affine head | 0.5743 | 0.1354 | 0.5409 | 0.3262 |
| Likelihood-consistent totalVI + D1 head | 0.5740 | 0.1359 | **0.5405** | 0.3262 |

Restoring the learned efficiency reduced RMSLE by 0.4951 relative to the helper output (95% CI −0.4957 to −0.4944; 16/16 paired wins; paired t-test p=7.1×10⁻⁴¹).

The D1 affine head did **not** improve correctly read totalVI. It increased RMSLE by 0.0093 (95% CI +0.0081 to +0.0105; 0/16 wins; p=4.2×10⁻¹¹). Applying the head after efficiency restoration also increased RMSLE by 0.0090 (95% CI +0.0079 to +0.0102; 0/16 wins; p=6.7×10⁻¹¹).

### External baselines

| Method | Mean protein RMSLE | Spearman | Within-cell-type Spearman |
|---|---:|---:|---:|
| **Likelihood-consistent totalVI** | **0.5650** | **0.3262** | 0.1678 |
| D1 affine head | 0.5743 | 0.3262 | 0.1678 |
| Official Seurat v3 prediction | 0.6054 | 0.3032 | 0.1338 |
| RNA ridge + D1 calibration | 0.6370 | 0.3218 | **0.1689** |
| RNA kNN + D1 calibration | 0.7109 | 0.2697 | 0.1159 |

Likelihood-consistent totalVI restores the ordering reported by the paper: totalVI outperforms the authors' published Seurat target prediction on the aggregate RMSLE used here.

## Conclusion

The proposed D1 affine-head improvement claim is rejected. Its apparent 45.8% advantage resulted from comparing a D1-fitted scale correction with a helper output that omitted a learned factor used by the scvi-tools 1.4.2 likelihood. Correctly read totalVI is already better than the affine head.

The useful result is a reproducibility finding: when evaluating protein imputation against observed counts in scvi-tools 1.4.2, predictions must include `log_per_batch_efficiency`; otherwise protein abundance is systematically inflated and method rankings can reverse. This does not establish that `get_normalized_expression()` is generally defective—the function returns normalized expression—but it is the wrong readout for this raw-count imputation benchmark unless the efficiency is restored.

Scale-free correlations were essentially unchanged, confirming that the error was in abundance scale rather than cell-level biological ordering.

## Reproducibility

```powershell
python src/evaluate_totalvix_paper.py --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041
python src/import_official_seurat.py
python src/compare_methods_calibrated.py --benchmark paper --skip-neural --calib-cells 0 --tag totalvix_baselines
python src/build_totalvix_benchmark_table.py
python src/plot_totalvix_paper_benchmark.py
```

Main outputs:

- `results/totalvix_paper_seed_summary.csv`
- `results/totalvix_paper_contrasts.csv`
- `results/totalvi_efficiency_factors.csv`
- `results/totalvi_efficiency_method_benchmark.csv`
- `results/totalvi_efficiency_vs_methods.csv`
- `results/figures/totalvix_paper_benchmark.png`

## Limitations

- Sixteen initializations were evaluated rather than the paper's 30.
- The benchmark contains one source-target mouse pair.
- The official Seurat target prediction is deterministic; R/Seurat is unavailable locally, so a D1-cross-fitted calibrated Seurat row was not generated.
- The finding is specific to the scvi-tools 1.4.2 readout and this count-scale benchmark. Other scvi-tools versions and downstream uses should be checked separately.
