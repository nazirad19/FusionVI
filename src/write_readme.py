"""Write the public project README from final benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiments = json.loads((ROOT / "results" / "fusionvi_experiments_summary.json").read_text())["experiments"]
    ridge = experiments[3]["rna_ridge_baseline"]
    controls = experiments[3]["confirmatory_controls"]
    cmeans = controls["mean_rmsle"]
    contrasts = {(r["candidate"], r["reference"]): r for r in controls["contrasts"]}
    same_width = contrasts[("FusionVI", "totalvi_w128")]
    param_match = contrasts[("FusionVI", "totalvi_pmatch")]
    availability = contrasts[("FusionVI", "totalvi_avail_w128")]
    indicator = contrasts[("totalvi_avail_w128", "totalvi_w128")]
    original = contrasts[("FusionVI", "totalVI")]
    n_seeds = same_width["n_seeds"]
    calibration = pd.read_csv(ROOT / "results" / "calibration_summary.csv").set_index(["arm", "readout"])
    calibration_contrasts = pd.read_csv(ROOT / "results" / "calibration_contrasts.csv")

    def cal(arm: str, readout: str, metric: str) -> float:
        return float(calibration.loc[(arm, readout), metric])

    def cal_contrast(metric: str) -> dict:
        row = calibration_contrasts[
            (calibration_contrasts["readout"] == "calibrated")
            & (calibration_contrasts["metric"] == metric)
            & (calibration_contrasts["reference"] == "totalvi")
        ].iloc[0]
        return row.to_dict()

    calibrated_rmsle = cal_contrast("rmsle")
    calibrated_residual = cal_contrast("residual_sd")
    readme = f"""# FusionVI

FusionVI is an experimental encoder variant evaluated on the published
totalVI missing-protein benchmark, with explicit capacity and missingness
controls added after reanalysis.

## Question

Can RNA in a new CITE-seq batch recover an entire unmeasured surface-protein
panel, and does a modality-specific fusion encoder improve that recovery?

This problem matters for therapeutic biomarker studies because surface proteins
define immune populations and drug targets, yet antibody panels may fail or
differ across experiments.

## Controlled benchmark

The repository uses the official SLN111 object released with the totalVI paper:

- 16,828 cells
- 4,005 highly variable genes
- 110 biological proteins
- SLN111-D1 as the complete RNA-protein reference batch
- SLN111-D2 as the target batch with every protein hidden during training

The comparison fixes the data, 20-dimensional latent space, native totalVI
decoder and likelihoods, optimizer, validation split, training budget and
posterior prediction procedure. The original encoder variants differ in width
and library-size estimation; the new control arms isolate those differences.

### Models

- **totalVI:** the published joint RNA-protein encoder.
- **FusionVI:** separate RNA and protein branches combined by a learned
  cell-level gate, with an explicit RNA-only route when proteins are absent.

## Main result

Across {n_seeds} paired random initializations, FusionVI reached mean
per-protein RMSLE **{cmeans['FusionVI']:.4f}** using the original
`log1p(E[y])` readout. It was lower than:

- the same-width joint encoder by **{abs(same_width['rmsle_diff']):.4f}**
  (95% CI {same_width['ci95_low']:+.4f} to {same_width['ci95_high']:+.4f};
  {same_width['seeds_candidate_better']}/16 seed wins);
- the parameter-matched joint encoder by **{abs(param_match['rmsle_diff']):.4f}**
  (95% CI {param_match['ci95_low']:+.4f} to {param_match['ci95_high']:+.4f};
  {param_match['seeds_candidate_better']}/16 wins); and
- the missingness-aware joint encoder by **{abs(availability['rmsle_diff']):.4f}**
  (95% CI {availability['ci95_low']:+.4f} to {availability['ci95_high']:+.4f};
  {availability['seeds_candidate_better']}/16 wins).

Against the published totalVI configuration, the difference was
{original['rmsle_diff']:+.4f} (95% CI {original['ci95_low']:+.4f} to
{original['ci95_high']:+.4f}; 14/16 wins). Adding a missing-panel indicator to
the same-width joint encoder did not improve RMSLE ({indicator['rmsle_diff']:+.4f};
p={indicator['p_t']:.2f}).

That readout does not target the same scale as RMSLE. Every checkpoint was
therefore re-scored without retraining using posterior `E[log1p y]` and a
per-protein affine calibration fitted only on D1 RNA-only predictions:

| Readout | totalVI | FusionVI | FusionVI − totalVI |
|---|---:|---:|---:|
| Original `log1p(E[y])` | {cal('totalvi', 'log_mean', 'rmsle'):.4f} | {cal('fusionvi', 'log_mean', 'rmsle'):.4f} | {original['rmsle_diff']:+.4f} |
| Posterior `E[log1p y]` | {cal('totalvi', 'pred_log', 'rmsle'):.4f} | {cal('fusionvi', 'pred_log', 'rmsle'):.4f} | {cal('fusionvi', 'pred_log', 'rmsle')-cal('totalvi', 'pred_log', 'rmsle'):+.4f} |
| D1-only calibrated | **{cal('totalvi', 'calibrated', 'rmsle'):.4f}** | {cal('fusionvi', 'calibrated', 'rmsle'):.4f} | {calibrated_rmsle['fusionvi_minus_ref']:+.4f} |
| RNA SVD-ridge | {cal('rna_ridge', 'ridge', 'rmsle'):.4f} | — | — |

After calibration, FusionVI's RMSLE was numerically higher by
{calibrated_rmsle['fusionvi_minus_ref']:+.4f} (95% CI
{calibrated_rmsle['ci95_low']:+.4f} to {calibrated_rmsle['ci95_high']:+.4f};
Holm p={calibrated_rmsle['p_holm_within_readout_metric']:.3f}). FusionVI also
had higher residual error ({calibrated_residual['fusionvi_minus_ref']:+.4f})
and lower within-cell-type Spearman ({cal('fusionvi', 'calibrated', 'within_celltype_spearman'):.4f}
versus {cal('totalvi', 'calibrated', 'within_celltype_spearman'):.4f}). Marker
AUROC was saturated for both models at approximately 0.99.

The evidence therefore supports a calibration result: FusionVI's original
RMSLE lead came mostly from output scale, not stronger recovery of biological
variation. D1 calibration also moved both neural models below the widened RNA
ridge RMSLE of {cal('rna_ridge', 'ridge', 'rmsle'):.4f}, showing that the old
ridge-versus-neural gap was likewise dominated by scale.

The paper used 30 initializations. This repository records that protocol but
runs {n_seeds} paired seeds for the course benchmark. The result supports a bounded
algorithmic comparison on one source-target batch pair; it does not establish
clinical or population-level biological generalization.

![Scale-matched complete-panel evaluation](results/figures/calibration_benchmark.png)

## Reproduce

The executed environment used Python 3.12, scvi-tools 1.4.2 and PyTorch 2.8.0
with one CUDA GPU.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
.\\run_paper_benchmark.ps1
.\\run_controls.ps1 -Tier 1
python src\\rna_baseline_paper_benchmark.py
.\\run_calibration.ps1
python -m unittest discover -s tests -v
```

Completed runs are detected and skipped. The pipeline downloads and verifies
the source object, creates the paper-aligned missing-panel split, trains both
models under paired seeds, evaluates every protein and regenerates all compact
results, figures and reports.

## Repository structure

- `config/paper_benchmark.yaml`: fixed benchmark settings and paired seeds.
- `src/prepare_paper_benchmark.py`: creates the missing-panel analysis object.
- `src/train_paper_benchmark.py`: trains one model for one seed.
- `src/fusionvi.py`: totalVI-compatible, missingness-aware dual-branch encoder.
- `src/evaluate_paper_benchmark.py`: aggregates metrics and figures.
- `src/rna_baseline_paper_benchmark.py`: source-only tuned RNA ridge baseline.
- `src/evaluate_biological_metrics.py`: foreground, within-cell-type and marker-AUROC metrics.
- `src/evaluate_calibration.py`: scale-matched readouts, bias/residual decomposition and biological metrics.
- `src/plot_calibration.py`: final calibration benchmark figure.
- `src/plot_heldout_experiments.py`: neutral summary figures for Experiments 1--3.
- `src/stats_paper_benchmark.py`: seed-level confidence intervals and exact tests.
- `src/benchmark_arms.py`: capacity, missingness and gate control encoders.
- `run_controls.ps1`: resumable seed-major control benchmark.
- `CONTROLS.md`: control rationale and interpretation limits.
- `TECHNICAL_REPORT.md`: concise GitHub-readable report.
- `deliverables/`: final PowerPoint and Word report.
- `results/`: compact per-seed and per-protein results.

Raw data, model weights and regenerated intermediate arrays are excluded from
Git.

## Reference

Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell
multi-omic data with totalVI. *Nature Methods*. 2021;18:272-282.
https://doi.org/10.1038/s41592-020-01050-x
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
