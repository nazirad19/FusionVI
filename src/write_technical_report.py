"""Write the concise, GitHub-readable technical report from final metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    headline = json.loads((RESULTS / "paper_benchmark_headline.json").read_text())
    seeds = pd.read_csv(RESULTS / "paper_benchmark_seed_summary.csv")
    seed_table = seeds.pivot(index="seed", columns="model", values="mean_rmsle").reset_index()
    total = headline["totalvi_mean_rmsle"]
    fusion = headline["fusionvi_mean_rmsle"]
    delta = headline["fusionvi_minus_totalvi_rmsle"]
    relative = 100.0 * (total - fusion) / total
    n_seeds = headline["completed_fusionvi_initializations"]
    if delta < -0.005:
        primary_statement = f"FusionVI reduced mean RMSLE by {abs(delta):.4f} ({relative:.2f}%)."
    elif delta > 0.005:
        primary_statement = f"FusionVI increased mean RMSLE by {delta:.4f} ({abs(relative):.2f}%)."
    else:
        primary_statement = (
            f"The models differed by only {abs(delta):.4f} mean RMSLE (FusionVI minus totalVI {delta:+.4f}), "
            "while FusionVI used 35% fewer trainable parameters."
        )

    seed_rows = "\n".join(
        f"| {int(row.seed)} | {row.totalVI:.4f} | {row.FusionVI:.4f} | {row.FusionVI - row.totalVI:+.4f} |"
        for row in seed_table.itertuples(index=False)
    )
    report = f"""# FusionVI technical report

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

Both models use learning rate 0.004, batch size 256, a maximum of 500 epochs, validation-based early stopping with patience 45 and 25 posterior samples for prediction. We executed {n_seeds} paired random initializations. The paper used 30 initializations, so this is a course-scale paper-aligned reproduction rather than an exact replication of its uncertainty analysis.

## Evaluation

The primary metric is per-protein root mean squared log error (RMSLE), matching the metric described for the paper experiment. Lower values are better. Secondary metrics are mean absolute error on log1p abundance, raw-count MAE, Spearman correlation and Pearson correlation on log1p abundance.

Proteins and random seeds are algorithmic benchmark units. They are not independent biological replicates, so the paired protein test is descriptive evidence about this dataset rather than population-level inference.

## Results

{primary_statement}

| Metric | totalVI | FusionVI | FusionVI minus totalVI |
|---|---:|---:|---:|
| RMSLE, primary | {f(total)} | {f(fusion)} | {delta:+.4f} |
| MAE, log1p | {f(headline['totalvi_mean_mae_log1p'])} | {f(headline['fusionvi_mean_mae_log1p'])} | {headline['fusionvi_minus_totalvi_mae_log1p']:+.4f} |
| MAE, raw abundance | {f(headline['totalvi_mean_mae_raw'])} | {f(headline['fusionvi_mean_mae_raw'])} | {headline['fusionvi_minus_totalvi_mae_raw']:+.4f} |
| Spearman correlation | {f(headline['totalvi_mean_spearman'])} | {f(headline['fusionvi_mean_spearman'])} | {headline['fusionvi_minus_totalvi_spearman']:+.4f} |
| Pearson correlation, log1p | {f(headline['totalvi_mean_pearson_log1p'])} | {f(headline['fusionvi_mean_pearson_log1p'])} | {headline['fusionvi_minus_totalvi_pearson_log1p']:+.4f} |

FusionVI had lower RMSLE for {headline['proteins_fusionvi_better']} of {headline['proteins_compared']} proteins. The paired two-sided Wilcoxon p-value across the 110 per-protein mean errors was {headline['protein_level_paired_wilcoxon_p']:.4g}.

| Seed | totalVI RMSLE | FusionVI RMSLE | Difference |
|---:|---:|---:|---:|
{seed_rows}

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
"""
    (ROOT / "TECHNICAL_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
