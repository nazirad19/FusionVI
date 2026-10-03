# FusionVI technical report

## Question

When an entire antibody panel is unavailable in a new CITE-seq batch, can a modality-specific fusion encoder recover immune surface-protein abundance more accurately than the published totalVI architecture?

This matters for therapeutic discovery because immune target and pharmacodynamic-marker panels are often incomplete across studies. Recovering an unmeasured surface panel can support dataset harmonization and hypothesis generation, although it does not replace experimental validation.

## Dataset and paper experiment

We used the processed SLN111 spleen and lymph-node CITE-seq object released with the totalVI paper (Gayoso et al., 2021; GSE150599). The benchmark follows the paper's Figure 3 missing-protein design:

- 16,828 cells, 4,005 selected genes and 110 non-hashtag proteins.
- SLN111-D1 contains RNA and proteins for 9,264 cells.
- SLN111-D2 contains RNA for 7,564 cells; all 110 proteins are hidden during training and retained only as evaluation truth.
- Predictions for D2 are decoded in the D1 batch to harmonize the batches.

## Models

**totalVI baseline.** The current scvi-tools implementation uses the paper's joint RNA-protein encoder, 20-dimensional latent state, negative-binomial gene likelihood and background-aware protein likelihood.

**FusionVI.** FusionVI changes only the encoder. Separate RNA and protein branches feed a learned cell-specific gate; when the protein panel is absent, an explicit availability rule routes the cell through the RNA branch. The same latent state and native totalVI decoder are retained. The benchmark version has 2.97 million trainable parameters, compared with 4.57 million for totalVI.

Both models use learning rate 0.004, batch size 256, a maximum of 500 epochs, validation-based early stopping with patience 45 and 25 posterior samples for prediction. We executed 4 paired random initializations. The paper used 30 initializations, so this is a course-scale paper-aligned reproduction rather than an exact replication of its uncertainty analysis.

## Evaluation

The primary metric is per-protein root mean squared log error (RMSLE), matching the metric described for the paper experiment. Lower values are better. Secondary metrics are mean absolute error on log1p abundance, raw-count MAE, Spearman correlation and Pearson correlation on log1p abundance.

Proteins and random seeds are algorithmic benchmark units. They are not independent biological replicates, so the paired protein test is descriptive evidence about this dataset rather than population-level inference.

## Results

FusionVI reduced mean RMSLE by 0.0060 (0.57%).

| Metric | totalVI | FusionVI | FusionVI minus totalVI |
|---|---:|---:|---:|
| RMSLE, primary | 1.0605 | 1.0544 | -0.0060 |
| MAE, log1p | 0.9307 | 0.9242 | -0.0065 |
| MAE, raw abundance | 34.7149 | 34.1587 | -0.5562 |
| Spearman correlation | 0.3257 | 0.3248 | -0.0008 |
| Pearson correlation, log1p | 0.4553 | 0.4530 | -0.0024 |

FusionVI had lower RMSLE for 74 of 110 proteins. The paired two-sided Wilcoxon p-value across the 110 per-protein mean errors was 1.037e-05.

| Seed | totalVI RMSLE | FusionVI RMSLE | Difference |
|---:|---:|---:|---:|
| 2026 | 1.0639 | 1.0482 | -0.0157 |
| 2027 | 1.0602 | 1.0619 | +0.0018 |
| 2028 | 1.0587 | 1.0522 | -0.0065 |
| 2029 | 1.0590 | 1.0554 | -0.0037 |

![Paper-aligned benchmark](results/figures/paper_benchmark_totalvi_vs_fusionvi.png)

### Secondary marker-recovery extension

In a separate two-fold leave-one-mouse-out experiment, four markers were hidden together while the other 106 proteins remained visible. A targeted FusionVI readout increased mean Spearman correlation from 0.578 to 0.673 across CD20, CD28, CD4 and CD8a. This secondary task evaluates a complete supervised prediction pipeline and does not isolate the encoder effect.

## Interpretation

The paper-aligned benchmark directly tests missing-modality integration on the original totalVI dataset. Its primary result should determine any claim of superiority. The secondary four-marker experiment asks a narrower biological question and shows that multimodal context can preserve drug-relevant immune-cell identity when selected antibody measurements are missing.

## Limitations

- The paper-aligned benchmark uses one source batch and one target batch.
- Five initializations give a useful robustness check but do not reproduce the paper's 30-run uncertainty analysis.
- The current software stack reimplements the protocol with scvi-tools 1.4.2; it does not execute the paper's historical code unchanged.
- Protein imputation supports exploratory biomarker work. It cannot substitute for prospective antibody measurements or clinical validation.

## Reproducibility

Run `run_paper_benchmark.ps1` for the paper-aligned benchmark and `run_all.ps1` for the four-marker extension. The repository records the dataset checksum, environment versions, preprocessing rules, paired seeds and all compact result tables. Raw data, weights and regenerated intermediate arrays stay outside Git.

## References

1. Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nature Methods*. 2021;18:272-282. https://doi.org/10.1038/s41592-020-01050-x
2. totalVI reproducibility repository: https://github.com/YosefLab/totalVI_reproducibility
