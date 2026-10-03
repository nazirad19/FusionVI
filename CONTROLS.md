# Controls and seed-level statistics

Added to address the review of the consolidated technical report. Nothing in the
original training path changes for the `totalvi` and `fusionvi` arms: same
parameter counts (4,574,975 / 2,971,904) and same run folders, so existing
seeds 2026–2029 are reused.

## Already computed (no new training)

| File | What it shows |
|---|---|
| `results/paper_benchmark_seed_inference.json` | Exp 4 with the seed as the unit. RMSLE difference −0.0060, t 95% CI [−0.0176, 0.0056], p = 0.20, exact sign-flip p = 0.25 (minimum attainable with 4 seeds: 0.125). About 14 paired seeds are needed for 80% power at the observed effect. |
| `results/cross_mouse_baselines*.{csv,json}` | Exp 3 latent-free readouts. Protein context + matching transcript: mean Spearman 0.670, against 0.671 (totalVI-X) and 0.673 (FusionVI-X). The latent adds ≤ 0.003. The protein panel alone (0.669) beats both native decoders (0.578 / 0.589). |

With n = 4, report the t interval. The bootstrap interval is too narrow at that sample size.

## Where the 35% parameter saving comes from

| Encoder component | totalVI (256) | FusionVI (128) |
|---|---|---|
| z network | 1,121,536 (RNA+protein → 256 → 256) | 530,304 RNA + 31,744 protein + 16,513 gate |
| library-size network | 1,054,720 (a second RNA+protein → 256 → 256 net) | 0 (reuses the RNA branch) |
| decoder and everything else | 2,387,925 | 2,387,925 |

Most of the saving comes from dropping totalVI's separate library encoder, and
the rest from halving the width. The modality split itself saves little.

## New arms (`config/paper_benchmark.yaml` → `arms:`)

| Arm | Encoder | Params | Question |
|---|---|---|---|
| `totalvi_w128` | joint, width 128 | 3.47M | fusion vs joint at the **same width** |
| `totalvi_pmatch` | joint, width 70 | 2.97M | fusion vs joint at the **same parameter count** |
| `totalvi_avail_w128` | joint + missing-panel indicator and learned placeholder | 3.47M | is the gain just **handling missing input**? |
| `fusionvi_pmatch` | fusion, width 406 | 4.58M | fusion at totalVI's capacity |
| `fusionvi_fixedgate` | gate fixed at 0.5 | 2.97M | does **learning** the gate matter? |
| `fusionvi_rnaonly` | gate fixed at 1 | 2.97M | does the protein branch matter at all? |
| `totalvi_moddrop` | joint, width 256, 30% panel dropout | pending | modality dropout applied to the published totalVI configuration |
| `totalvi_w128_moddrop` | joint, width 128, 30% panel dropout | pending | same-width joint control trained to decode proteins from RNA-only latents |
| `fusionvi_moddrop` | gated fusion, width 128, 30% panel dropout | pending | does supervised RNA-only training improve the missing-panel task? |

Tier 1 holds the first three arms and is the minimum. Each FusionVI arm also writes
`gates.csv`, the per-cell RNA weight (1 = RNA only). Target-batch cells are 1 by construction.

```powershell
.\run_controls.ps1          # tier 1, 16 seeds, seed-major order
.\run_controls.ps1 -Tier 2  # plus the tier 2 arms
.\run_controls.ps1 -Tier 3  # plus paired modality-dropout arms
```

The modality-dropout arms hide the entire encoder-side protein panel for 30%
of measured D1 cells during training while leaving those cells' protein
likelihood active. Panel availability is taken from batch metadata rather than
inferred from a zero count sum.

`src/rna_baseline_paper_benchmark.py` fits a source-only tuned RNA ridge
baseline. `src/evaluate_biological_metrics.py` adds foreground RMSLE,
within-cell-type Spearman and source-thresholded CD4/CD8/CD19 AUROC.

`src/evaluate_controls.py` writes `control_contrasts.csv` (seed-paired RMSLE
differences, t CI, exact p, Holm across contrasts) and
`control_gate_by_celltype.csv`. `src/param_match.py` recomputes the matched widths.

All arms were smoke-tested on the local GPU (800 cells, 2 epochs). Real numbers
still require full-data paired runs.

## Experiments 1 and 2

The original Lawlor and Papalexi training source was not retained. The branch
does include the 75 saved Papalexi target-by-replicate effects needed to rerun
the target-cluster bootstrap. Lawlor cell-level predictions are unavailable.

```bash
python src/stats_heldout_experiments.py exp2 --csv results/papalexi_effects.csv   # target-cluster bootstrap CIs
python src/stats_heldout_experiments.py exp1 --csv lawlor_cells.csv       # RNA strata + leakage-safe discordance
python src/stats_heldout_experiments.py holm --p exp1=... exp2=... exp3=... exp4=...
```

## Other fixes

- The original `src/prepare_paper_benchmark.py` and
  `src/evaluate_paper_benchmark.py` are now included on the branch so both
  PowerShell runners work from a fresh clone.
- `src/evaluate.py` now labels its readout `FusionVI-X`. Regenerate the legacy
  output files before citing them.
