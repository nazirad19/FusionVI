# FusionVI-X

FusionVI-X tests whether targeted cross-modal learning can recover hidden
pharmacodynamic surface markers from paired single-cell RNA and protein data.
It asks:

> When immune stimulation produces disagreement between transcript and
> surface-protein evidence, can cross-modal context recover CD25, CD69 and
> HLA-DR in unseen donors more reliably than standard totalVI?

The study uses the Lawlor et al. PBMC CITE-seq experiment: 16,382 cells from 10
paired donors under baseline, LPS, or anti-CD3/CD28 stimulation, with 39 ADTs.

## Method

The study contains two linked experiments.

**Encoder ablation.** FusionVI replaces totalVI's joint encoder with an RNA
branch, a protein branch and a learned scalar gate for every cell. The decoder,
likelihoods, latent size, training objective, preprocessing, seed and donor
splits remain fixed.

**Targeted cross-modal readout.** FusionVI-X predicts each hidden surface marker
from the FusionVI latent state, the remaining 36 ADTs and a prespecified
marker-matched RNA proxy. Ridge, histogram gradient boosting and Extra Trees are
candidate heads. Within each outer fold, the head family is selected by
three-fold grouped validation among the nine training donors, refit on those
nine donors, and evaluated once on the untouched tenth donor. Standard totalVI
receives the identical readout as a control (`totalVI-X`).

CD25, CD69 and HLA-DR are zeroed at encoder input and never used as readout
features. Their measured values are available only as training targets in the
nine training donors and as evaluation targets in the held-out donor. CD80 was
measured by external flow cytometry in the source paper but is absent from the
released 39-ADT matrix.

## Evaluation design

- **10-fold leave-one-donor-out validation**: each donor is the test set once.
- **Nested model selection**: readout family selected using training donors only.
- **Direct responses**: anti-CD3/CD28 T cells and LPS monocytes.
- **Held-out markers**: CD25/CD69 in T cells and HLA-DR in monocytes.
- **Discordance challenge**: marker RNA and measured protein percentile ranks
  differ by at least 0.50 within donor and relevant cell population.
- **Ablations**: RNA proxy, remaining ADTs, RNA+ADT context, totalVI-X and
  FusionVI-X.
- **Statistics**: one value per held-out donor, paired Wilcoxon tests, bootstrap
  confidence intervals and Benjamini-Hochberg correction.

## Executed results

All 20 neural folds completed at 40 epochs. Broad stimulation labels were already
at ceiling: standard totalVI achieved mean donor-held-out ROC AUCs of 0.995 to
1.000. The scalar gate learned biological context but did not reliably improve
marker reconstruction by itself.

The nested cross-modal readout produced consistent positive results relative to
the standard totalVI decoder:

| Endpoint | totalVI | FusionVI-X | Paired difference | BH q |
|---|---:|---:|---:|---:|
| CD25 global Spearman | 0.779 | 0.825 | +0.046 | 0.0039 |
| CD69 global Spearman | 0.822 | 0.873 | +0.051 | 0.0039 |
| HLA-DR global Spearman | 0.404 | 0.756 | +0.352 | 0.0039 |
| HLA-DR discordant Spearman | -0.175 | 0.637 | +0.812 | 0.0117 |

Discordant CD25 and CD69 improvements were positive but not statistically
reliable. The ablation gives the biological interpretation: a marker-matched RNA
proxy alone became anticorrelated with surface abundance in discordant cells,
whereas the remaining surface-protein panel restored predictive signal. The
successful contribution is therefore targeted cross-modal biomarker readout;
the scalar gate is retained as an informative negative architecture ablation.

## Reproduce

The executed environment used Python 3.12, scvi-tools 1.4.2, PyTorch 2.8.0 and
one CUDA GPU. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\run_all.ps1
```

`run_all.ps1` downloads checksum-verified matrices, prepares the AnnData object,
runs both neural models, evaluates classical baselines, fits the nested
cross-modal readout and regenerates every result figure.

## Repository layout

- `config/default.yaml` — fixed experiment configuration.
- `src/download_data.py` — HCA downloads with expected checksums.
- `src/prepare_data.py` — annotation matching, count preservation and
  lane-aware highly-variable-gene selection.
- `src/fusion_encoder.py` — masked totalVI and gated dual-branch encoders.
- `src/train_fold.py` — one deterministic donor-held-out neural fold.
- `src/classical_baselines.py` — donor-held-out PCA baselines.
- `src/evaluate.py` — biological metrics and donor-level inference.
- `src/cross_modal_head.py` — nested donor-safe cross-modal readout.
- `src/make_cross_modal_figures.py` — positive-result and ablation figures.
- `results/` — compact metrics, selected head families and figures.

Raw matrices, processed AnnData, model weights, per-cell fold outputs and
cross-modal cell-level predictions are regenerated and excluded from Git.

## Scope

This is an exploratory pharmacodynamic biomarker study at one dose and one
24-hour timepoint. It demonstrates cross-donor recovery of molecular response
markers; it does not establish causality, dose-response behaviour or clinical
utility.
