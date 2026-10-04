# Controls and seed-level statistics

The Tier 1 control benchmark is complete. All neural models use the same SLN111
source-target split, latent size, totalVI decoder and likelihoods, optimizer,
500-epoch budget, and posterior prediction procedure. The random seed is the
replication unit.

## Completed result

Sixteen paired seeds were run for FusionVI, the published totalVI configuration,
and three joint-encoder controls.

| Candidate minus reference | Mean RMSLE difference | 95% t interval | Seed wins | Holm p |
|---|---:|---:|---:|---:|
| FusionVI - joint same width | -0.0106 | [-0.0135, -0.0077] | 15/16 | 3.38e-6 |
| FusionVI - joint parameter matched | -0.0189 | [-0.0216, -0.0162] | 16/16 | 7.91e-10 |
| FusionVI - joint plus availability | -0.0112 | [-0.0143, -0.0081] | 15/16 | 3.38e-6 |
| Joint plus availability - joint same width | +0.0006 | [-0.0013, +0.0025] | 7/16 | 0.514 |
| FusionVI - published totalVI | -0.0062 | [-0.0086, -0.0038] | 14/16 | not in control family |

Lower RMSLE is better. FusionVI's small advantage survives controls for encoder
width, trainable parameter count, and explicit missing-panel information. The
availability indicator alone did not help. A source-only RNA SVD-ridge baseline
still achieved RMSLE 0.6379, far below every neural model (1.0539-1.0728), so
the practical conclusion remains limited to this encoder comparison.

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

## Scale-matched and calibrated re-evaluation (no retraining)

RMSLE is minimized by predicting E[log1p y]. The neural arms were scored on
log1p(E[y]), which is biased upward for over-dispersed counts, while the RNA
ridge is fit on log1p(y) directly. That alone can create large RMSLE gaps
between models that rank cells equally well (ridge Spearman 0.322 vs FusionVI
0.326, yet RMSLE 0.638 vs 1.054). `src/evaluate_calibration.py` reloads each
saved model and scores D2 three ways:

| Readout | Definition |
|---|---|
| `log_mean` | log1p(E[y]) with background, source-batch decoding (original metric; reproduces scvi's `get_normalized_expression` within Monte Carlo noise) |
| `pred_log` | E[log1p y]: mean of log1p over posterior predictive draws from the protein NB mixture |
| `calibrated` | per-protein affine map of `log_mean`, fitted on D1 cells pushed through the same RNA-only encoder route as D2; no D2 truth used |

Every row reports RMSLE with its decomposition RMSLE² = bias² + residual_sd²,
plus Spearman and Pearson. The ridge baseline is re-scored on the same cells
(alpha grid extended to 1e5). Seed-paired FusionVI-minus-control contrasts
are reported for every readout × metric.

```powershell
.\run_calibration.ps1                                   # tier 0+1 arms, all control seeds
.\run_calibration.ps1 --arms totalvi fusionvi --seeds 2026 2027
```

The completed 16-seed result follows that second pattern. FusionVI's original
RMSLE difference versus totalVI was -0.0062, but after D1-only calibration it
was +0.0085 (95% CI -0.0003 to +0.0173; Holm p=0.218). FusionVI also had higher
residual error (+0.0065) and lower within-cell-type Spearman (0.1640 versus
0.1681). Marker AUROC was effectively saturated for both models (0.9903 versus
0.9909). The defensible conclusion is therefore that FusionVI changed output
scale without adding detectable information about D2 protein variation.

The re-evaluation also writes compact within-cell-type summaries and
source-thresholded CD4/CD8/CD19 AUROCs. Memory scales with posterior draws ×
batch × genes; lower `batch_size` in `predict` if a GPU runs out of memory.
