"""Write the public project README from final benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    h = json.loads((ROOT / "results" / "paper_benchmark_headline.json").read_text())
    inference = json.loads((ROOT / "results" / "paper_benchmark_seed_inference.json").read_text())["metrics"]["rmsle"]
    total = h["totalvi_mean_rmsle"]
    fusion = h["fusionvi_mean_rmsle"]
    delta = h["fusionvi_minus_totalvi_rmsle"]
    relation = "lower" if delta < 0 else "higher"
    params = {r["model"]: r["trainable_parameters"] for r in h["completed_runs"]}
    reduction = 100 * (params["totalVI"] - params["FusionVI"]) / params["totalVI"]
    n_seeds = h["completed_fusionvi_initializations"]
    experiments = json.loads((ROOT / "results" / "fusionvi_experiments_summary.json").read_text())["experiments"]
    ridge = experiments[3]["rna_ridge_baseline"]
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

Across {n_seeds} paired random initializations, mean per-protein RMSLE was
**{total:.4f} for totalVI** and **{fusion:.4f} for FusionVI**. FusionVI was
**{abs(delta):.4f} {relation}** on the primary metric and used
**{reduction:.1f}% fewer trainable parameters**. FusionVI had lower mean RMSLE
for **{h['proteins_fusionvi_better']} of {h['proteins_compared']} proteins**.

The seed-paired RMSLE difference was {inference['mean_difference']:+.4f}
(95% t interval {inference['t_ci95'][0]:+.4f} to {inference['t_ci95'][1]:+.4f};
p={inference['paired_t_p']:.2f}). The interval includes zero. The earlier
protein-level p-value is descriptive because proteins are repeated outcomes
within each initialization.

A source-only RNA baseline using a 128-component SVD and multi-output ridge
reached mean protein RMSLE **{ridge['mean_protein_rmsle']:.4f}**, substantially
below both neural models. Its CD4/CD8/CD19 AUROCs exceeded 0.97, but its mean
within-cell-type Spearman was only {ridge['mean_within_celltype_spearman']:.3f}.
Aggregate error and marker separation can therefore look strong without
recovering subtle within-cell-state variation.

The original comparison is also capacity-confounded: totalVI uses a 256-unit
joint encoder and a separate library encoder, while FusionVI uses 128-unit
branches and reuses the RNA branch for library size. The parameter reduction
therefore cannot be attributed to modality fusion. `run_controls.ps1` adds
same-width, parameter-matched and missing-panel-aware joint controls.
Tier 3 adds paired totalVI and FusionVI arms with 30% supervised whole-panel
dropout, so the protein decoder learns from RNA-only latents under a fair
comparison.

The paper used 30 initializations. This repository records that protocol but
runs {n_seeds} paired seeds for the course benchmark. The result supports a bounded
algorithmic comparison on one source-target batch pair; it does not establish
clinical or population-level biological generalization.

![Paper-aligned totalVI versus FusionVI benchmark](results/figures/paper_benchmark_totalvi_vs_fusionvi.png)

## Reproduce

The executed environment used Python 3.12, scvi-tools 1.4.2 and PyTorch 2.8.0
with one CUDA GPU.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
.\\run_paper_benchmark.ps1
.\\run_controls.ps1 -Tier 1
python src\\rna_baseline_paper_benchmark.py
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
