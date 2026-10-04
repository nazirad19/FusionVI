# Controls and seed-level statistics

The Tier 1 control benchmark is complete. All neural models use the same SLN111
source-target split, latent size, totalVI decoder and likelihoods, optimizer,
500-epoch budget, and posterior prediction procedure. The random seed is the
replication unit.

## Historical helper-output audit

Sixteen paired seeds were run for FusionVI, the published totalVI configuration,
and three joint-encoder controls.

| Candidate minus reference | Mean RMSLE difference | 95% t interval | Seed wins | Holm p |
|---|---:|---:|---:|---:|
| FusionVI - joint same width | -0.0106 | [-0.0135, -0.0077] | 15/16 | 3.38e-6 |
| FusionVI - joint parameter matched | -0.0189 | [-0.0216, -0.0162] | 16/16 | 7.91e-10 |
| FusionVI - joint plus availability | -0.0112 | [-0.0143, -0.0081] | 15/16 | 3.38e-6 |
| Joint plus availability - joint same width | +0.0006 | [-0.0013, +0.0025] | 7/16 | 0.514 |
| FusionVI - published totalVI | -0.0062 | [-0.0086, -0.0038] | 14/16 | not in control family |

These values used the efficiency-omitting `get_normalized_expression()` helper
output and are retained only to document the original analysis. They are not a
valid count-scale encoder comparison and must not be cited as the Experiment 4
result. The likelihood-consistent comparison below supersedes them.

## Where the original parameter saving comes from

| Encoder component | totalVI width 256 | FusionVI width 128 |
|---|---:|---:|
| z network | 1,121,536 | 530,304 RNA + 31,744 protein + 16,513 gate |
| library-size network | 1,054,720 | 0; reuses the RNA branch |
| decoder and everything else | 2,387,925 | 2,387,925 |

Most of the 35% parameter difference comes from dropping totalVI's separate
library encoder and halving the hidden width. The matched controls were required
to isolate the encoder design.

## Arms

| Arm | Encoder | Status | Question |
|---|---|---|---|
| `totalvi_w128` | joint, width 128 | complete, 16 seeds | FusionVI versus joint at the same width |
| `totalvi_pmatch` | joint, width 70 | complete, 16 seeds | FusionVI versus joint at the same parameter count |
| `totalvi_avail_w128` | joint plus availability indicator, width 128 | complete, 16 seeds | Is the gain only explicit missing-input handling? |
| `fusionvi_pmatch` | fusion, width 406 | not run | FusionVI at totalVI's capacity |
| `fusionvi_fixedgate` | gate fixed at 0.5 | not run | Does learning the gate help? |
| `fusionvi_rnaonly` | gate fixed at 1 | not run | Does the protein branch help on source cells? |
| `totalvi_moddrop` | joint, width 256, 30% panel dropout | not run | Published totalVI with supervised missing panels |
| `totalvi_w128_moddrop` | joint, width 128, 30% panel dropout | not run | Same-width joint encoder with supervised missing panels |
| `fusionvi_moddrop` | fusion, width 128, 30% panel dropout | not run | Does modality dropout improve the target task? |

```powershell
.\run_controls.ps1          # Tier 1, 16 seeds, completed runs are skipped
.\run_controls.ps1 -Tier 2  # add capacity and gate ablations
.\run_controls.ps1 -Tier 3  # add modality-dropout arms
```

`src/evaluate_controls.py` writes the full per-protein table, paired contrasts,
gate summaries, the control figure, and synchronizes Experiment 4 in the
consolidated metrics file. `src/param_match.py` recomputes matched widths.

## Reproducibility scope

- `src/prepare_paper_benchmark.py` and `src/evaluate_paper_benchmark.py` are
  versioned, so both PowerShell runners work from a fresh clone.
- The original Lawlor and Papalexi training source was not retained. The branch
  includes the 75 Papalexi target-by-replicate effects needed to rerun the
  target-cluster bootstrap. Lawlor cell-level predictions are unavailable.
- `src/evaluate.py` labels its supervised readout `FusionVI-X`; native decoder
  and cross-modal readout results are not pooled.

## Likelihood-consistent re-evaluation (no retraining)

scvi-tools 1.4.2 learns a per-protein, per-batch efficiency used by the protein
likelihood. The normalized-expression helper omits this factor, so
`src/evaluate_calibration.py` reloads every checkpoint and scores D2 four ways:

| Readout | Definition |
|---|---|
| `likelihood_mean` | **Primary:** log1p(E[y]) from `py_norm_`, including the learned efficiency |
| `likelihood_pred_log` | E[log1p y] from posterior predictive draws using `py_norm_` |
| `calibrated` | D1 affine head applied to `likelihood_mean`; no D2 truth used |
| `helper_mean` | Historical helper output from `py_`; efficiency omitted |

Every row reports RMSLE with its decomposition RMSLE² = bias² + residual_sd²,
plus Spearman and Pearson. The ridge baseline is re-scored on the same cells
(alpha grid extended to 1e5). Seed-paired FusionVI-minus-control contrasts
are reported for every readout × metric.

```powershell
.\run_calibration.ps1                                   # tier 0+1 arms, all control seeds
.\run_calibration.ps1 --arms totalvi fusionvi --seeds 2026 2027
```

The primary Experiment 4 claim comes from the seed-paired `likelihood_mean`
contrasts. A positive rescaling cannot change per-protein Spearman, so a win in
RMSLE alone supports improved count reconstruction rather than improved
biological ranking.

The re-evaluation also writes compact within-cell-type summaries and
source-thresholded CD4/CD8/CD19 AUROCs. Memory scales with posterior draws ×
batch × genes; lower `batch_size` in `predict` if a GPU runs out of memory.

## Focused totalVI readout audit

The final paper-benchmark analysis tests whether the apparent totalVI-X gain
survives after restoring totalVI's learned protein-efficiency factor. It
compares likelihood-consistent totalVI with the former affine totalVI-X
readout, the totalVI authors' published Seurat v3 target prediction, and
D1-calibrated RNA ridge and kNN baselines. FusionVI is excluded from this
focused analysis.

```powershell
python src/import_official_seurat.py
python src/compare_methods_calibrated.py --benchmark paper --skip-neural --calib-cells 0 --tag totalvix_baselines
python src/build_totalvix_benchmark_table.py
```

The headline outputs are
`results/totalvi_efficiency_method_benchmark.csv`,
`results/totalvi_efficiency_vs_methods.csv` and
`results/totalvix_paper_contrasts.csv`. The official Seurat result can be
scored raw without R. Generating a D1-calibrated Seurat row still requires R
with Seurat because it needs five-fold source predictions.

### Learned protein efficiency audit

scvi-tools 1.4.2 learns `log_per_batch_efficiency` and multiplies protein
rates by its exponential inside the reconstruction likelihood. Its
`get_normalized_expression()` implementation reads the unscaled `py_` rates,
whereas the likelihood uses `py_norm_`. The `likelihood_mean` readout restores this
model-owned factor without fitting a head or using target protein labels.
The 16-checkpoint audit found that likelihood-consistent totalVI reached 0.5650
RMSLE, compared with 1.0601 for the helper output and 0.5743 for the D1 affine
head. The head therefore does not improve correctly read totalVI; its apparent
gain reconstructed a model-owned scale factor. The name `totalVI-X` is reserved
for the cross-modal ridge readout in Experiments 1–3.
