"""Write the consolidated Markdown report for all completed FusionVI experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "fusionvi_experiments_summary.json"
OUT = ROOT / "TECHNICAL_REPORT.md"


def main() -> None:
    lawlor, papalexi, targeted, paper = json.loads(SUMMARY.read_text())["experiments"]
    pm = papalexi["metrics"]
    pb = papalexi["target_cluster_bootstrap"]
    pt = papalexi["paired_target_test"]
    ridge = paper["rna_ridge_baseline"]
    relative = 100 * (paper["totalvi_mean_rmsle"] - paper["fusionvi_mean_rmsle"]) / paper["totalvi_mean_rmsle"]
    parameter_reduction = 100 * (paper["totalvi_parameters"] - paper["fusionvi_parameters"]) / paper["totalvi_parameters"]

    text = f"""# FusionVI technical report

## Project question

Can modality-specific fusion improve recovery of therapeutic surface biomarkers when protein measurements are missing, discordant with RNA, or observed under an unseen perturbation?

FusionVI replaces totalVI's joint encoder with separate RNA and protein branches and a cell-level gate while retaining the totalVI decoder and likelihoods. The latest missing-panel version explicitly routes cells through the RNA branch when the complete protein panel is absent. Four experiments test different forms of generalization. Together, they show useful multimodal signal in therapeutic biomarker tasks, while the specific advantage of the FusionVI encoder remains uncertain after accounting for seed variation, measured protein context and model capacity.

## Experiment overview

| Experiment | Dataset and held-out unit | Biological question | Main result |
|---|---|---|---|
| 1. Donor-held-out activation | Lawlor PBMC CITE-seq; 10 donors | Can hidden CD25, CD69 and HLA-DR be recovered in an unseen donor, including RNA-protein-discordant cells? | Native encoders were close; a donor-safe cross-modal readout provided most of the gain. |
| 2. Unseen perturbations | Papalexi ECCITE-seq; 25 CRISPR targets | Given a held-out perturbation cell's RNA and three measured proteins, can its hidden PD-L1 response be recovered? | FusionVI-X achieved effect Spearman {pm['fusionvi_xmodal']['effect_spearman']:.3f} and 84.0% direction accuracy. |
| 3. Targeted cross-mouse transfer | Original totalVI SLN111; 2 mice | Can four hidden immune markers be transferred to an unseen mouse? | A latent-free protein-context baseline reached {targeted['protein_context_plus_rna_mean_spearman']:.3f}; adding either latent changed Spearman by no more than {max(targeted['totalvi_latent_increment'], targeted['fusionvi_latent_increment']):.3f}. |
| 4. Complete missing panel | totalVI Figure 3 SLN111 design; 4 paired seeds | Can RNA recover all 110 proteins in a batch with no protein input? | FusionVI's mean RMSLE was {abs(paper['fusionvi_minus_totalvi_rmsle']):.4f} lower, but a source-only RNA ridge baseline was substantially better than both neural models. |

## Method

**Published baseline.** totalVI models RNA with a negative-binomial likelihood and proteins with a background-foreground mixture. Its standard encoder jointly processes both modalities.

**FusionVI encoder.** Separate RNA and protein branches produce modality-specific hidden representations. A learned gate combines them before estimating the same latent distribution used by the native totalVI decoder. For the complete missing-panel benchmark, an availability rule forces an RNA-only route when all protein inputs are absent, preventing an all-zero placeholder panel and encoder biases from creating a spurious protein representation.

**FusionVI-X readout.** Experiments 1 to 3 also evaluated a nested, leakage-safe cross-modal readout. It predicts a hidden protein from the learned latent state, the remaining proteins and a prespecified matching transcript. The same readout applied to totalVI is called totalVI-X. Native-decoder and X-readout results are reported separately because they answer different questions.

## Experiment 1 Lawlor donor-held-out activation

The Lawlor PBMC CITE-seq dataset contains {lawlor['cells']:,} cells from 10 paired donors under baseline, LPS or anti-CD3/CD28 stimulation, with 4,000 genes and 39 antibody-derived tags. CD25 and CD69 were hidden in T cells and HLA-DR in monocytes. Every donor served once as the untouched outer test fold. Readout selection used only the other nine donors.

| Target | totalVI native | FusionVI native | totalVI-X | FusionVI-X |
|---|---:|---:|---:|---:|
| CD25 in T cells | {lawlor['native_totalvi_spearman'][0]:.3f} | {lawlor['native_fusionvi_spearman'][0]:.3f} | {lawlor['totalvi_xmodal_spearman'][0]:.3f} | {lawlor['fusionvi_xmodal_spearman'][0]:.3f} |
| CD69 in T cells | {lawlor['native_totalvi_spearman'][1]:.3f} | {lawlor['native_fusionvi_spearman'][1]:.3f} | {lawlor['totalvi_xmodal_spearman'][1]:.3f} | {lawlor['fusionvi_xmodal_spearman'][1]:.3f} |
| HLA-DR in monocytes | {lawlor['native_totalvi_spearman'][2]:.3f} | {lawlor['native_fusionvi_spearman'][2]:.3f} | {lawlor['totalvi_xmodal_spearman'][2]:.3f} | {lawlor['fusionvi_xmodal_spearman'][2]:.3f} |

The encoder change alone was a negative or near-null result. The cross-modal readout substantially improved hidden-marker recovery, but totalVI-X and FusionVI-X remained close. The ablation showed why multimodality matters: marker-matched RNA alone became misleading in discordant cells, whereas the remaining surface-protein panel restored useful signal.

![Lawlor hidden-marker recovery](results/figures/lawlor_marker_recovery.png)

## Experiment 2 Papalexi unseen CRISPR targets

The Papalexi ECCITE-seq screen contains {papalexi['cells']:,} IFN-gamma-treated THP-1 cells, 25 perturbed genes, three biological replicates and four surface proteins. Five outer folds kept every cell from a CRISPR target together, so each target was evaluated only after being excluded from training. At test time, the model still receives each perturbed cell's RNA profile and the other three measured proteins; only PD-L1 is hidden. The perturbation-target label is not an input. This is therefore cross-modal PD-L1 completion in cells carrying unseen perturbations, rather than de novo response prediction from target identity. Outcomes were calculated from 75 target-by-replicate effects.

| Model | Effect Spearman | Direction accuracy | Effect MAE |
|---|---:|---:|---:|
| CD274 RNA only | {pm['cd274_rna_only']['effect_spearman']:.3f} | {100*pm['cd274_rna_only']['direction_accuracy']:.1f}% | — |
| totalVI decoder | {pm['totalvi_decoder']['effect_spearman']:.3f} | {100*pm['totalvi_decoder']['direction_accuracy']:.1f}% | — |
| FusionVI decoder | {pm['fusionvi_decoder']['effect_spearman']:.3f} | {100*pm['fusionvi_decoder']['direction_accuracy']:.1f}% | — |
| totalVI-X | {pm['totalvi_xmodal']['effect_spearman']:.3f} | {100*pm['totalvi_xmodal']['direction_accuracy']:.1f}% | {pm['totalvi_xmodal']['effect_mae']:.3f} |
| FusionVI-X | **{pm['fusionvi_xmodal']['effect_spearman']:.3f}** | **{100*pm['fusionvi_xmodal']['direction_accuracy']:.1f}%** | **{pm['fusionvi_xmodal']['effect_mae']:.3f}** |

FusionVI-X reduced median per-target PD-L1 effect absolute error by {pt['median_absolute_error_reduction']:.4f} relative to totalVI-X after replicate averaging (paired Wilcoxon p={pt['p_value']:.3f}, 25 targets). In a 5,000-sample target-cluster bootstrap, FusionVI-X effect Spearman was {pb['models']['fusionvi_xmodal']['effect_spearman']['estimate']:.3f} (95% CI {pb['models']['fusionvi_xmodal']['effect_spearman']['ci95'][0]:.3f} to {pb['models']['fusionvi_xmodal']['effect_spearman']['ci95'][1]:.3f}); its difference from totalVI-X was {pb['contrast']['effect_spearman']['difference']:+.3f} (95% CI {pb['contrast']['effect_spearman']['ci95'][0]:+.3f} to {pb['contrast']['effect_spearman']['ci95'][1]:+.3f}). The MAE-difference interval included zero ({pb['contrast']['effect_mae']['ci95'][0]:+.3f} to {pb['contrast']['effect_mae']['ci95'][1]:+.3f}). It recovered the expected loss of PD-L1 after IFNGR1, IFNGR2, JAK2 and STAT1 perturbation and increased PD-L1 after CUL3 or BRD4 perturbation. It failed on CMTM6, where PD-L1 protein decreases despite slightly increased CD274 RNA. That failure is biologically informative because CMTM6 regulates PD-L1 stability after translation.

![Papalexi PD-L1 perturbation validation](results/figures/papalexi_pdl1_validation.png)

## Experiment 3 original totalVI targeted marker transfer

The official SLN111 object contains {targeted['cells']:,} mouse spleen and lymph-node cells, 4,000 genes and 110 proteins. CD20, CD28, CD4 and CD8a were masked together. Each model trained on one mouse and was evaluated on the other, then the direction was reversed.

| Readout | totalVI | FusionVI | Difference |
|---|---:|---:|---:|
| Native decoder mean Spearman | {targeted['native_totalvi_mean_spearman']:.3f} | {targeted['native_fusionvi_mean_spearman']:.3f} | {targeted['native_fusionvi_mean_spearman']-targeted['native_totalvi_mean_spearman']:+.3f} |
| Fixed cross-modal readout mean Spearman | {targeted['totalvi_xmodal_mean_spearman']:.3f} | {targeted['fusionvi_xmodal_mean_spearman']:.3f} | {targeted['fusionvi_xmodal_mean_spearman']-targeted['totalvi_xmodal_mean_spearman']:+.3f} |
| Latent-free protein context + RNA | {targeted['protein_context_plus_rna_mean_spearman']:.3f} | {targeted['protein_context_plus_rna_mean_spearman']:.3f} | shared baseline |

The native FusionVI decoder improved modestly, but the latent-free control changes the interpretation. The remaining 106 proteins plus the matching transcript reached {targeted['protein_context_plus_rna_mean_spearman']:.3f}; totalVI-X and FusionVI-X added only {targeted['totalvi_latent_increment']:.3f} and {targeted['fusionvi_latent_increment']:.3f}, respectively. Most cross-mouse performance came from measured protein context rather than either learned latent. Two mice support only a technical transfer conclusion.

![Original totalVI targeted marker recovery](results/figures/totalvi_original_marker_recovery.png)

## Experiment 4 paper-aligned complete missing panel

This experiment follows the totalVI paper's Figure 3 missing-protein test. SLN111-D1 retained RNA and all 110 proteins, while every D2 protein was hidden during training and preserved only for scoring. Both models used the same 20-dimensional latent space, native decoder and likelihoods, optimizer, split, 500-epoch budget and 25 posterior samples. Four paired seeds were run; the paper used 30.

| Metric | totalVI | FusionVI | Difference |
|---|---:|---:|---:|
| RMSLE primary | {paper['totalvi_mean_rmsle']:.4f} | **{paper['fusionvi_mean_rmsle']:.4f}** | {paper['fusionvi_minus_totalvi_rmsle']:+.4f} |
| Proteins with lower RMSLE | — | {paper['proteins_fusionvi_better']}/{paper['proteins_compared']} | — |
| Trainable parameters | {paper['totalvi_parameters']:,} | {paper['fusionvi_parameters']:,} | {parameter_reduction:.1f}% fewer |

The source-only RNA baseline fit a 128-component SVD and multi-output ridge on D1, selected its penalty using a D1 validation split, and then predicted D2. It reached mean protein RMSLE **{ridge['mean_protein_rmsle']:.4f}**, well below totalVI ({paper['totalvi_mean_rmsle']:.4f}) and FusionVI ({paper['fusionvi_mean_rmsle']:.4f}). Its source-defined marker AUROCs were {ridge['marker_auroc']['CD4']:.3f} for CD4, {ridge['marker_auroc']['CD8']:.3f} for CD8 and {ridge['marker_auroc']['CD19']:.3f} for CD19, while mean within-cell-type Spearman was only {ridge['mean_within_celltype_spearman']:.3f}. This contrast shows why aggregate error and marker separation can look strong while within-cell-state variation remains difficult.

FusionVI's average RMSLE was {relative:.2f}% lower, but the random initialization is the valid replication unit. The paired seed difference was {paper['fusionvi_minus_totalvi_rmsle']:+.4f} (95% CI {paper['seed_level_rmsle_ci95'][0]:+.4f} to {paper['seed_level_rmsle_ci95'][1]:+.4f}; paired t p={paper['seed_level_paired_t_p']:.2f}; exact sign-flip p={paper['seed_level_exact_p']:.2f}). Three of four seeds favored FusionVI, and approximately {paper['seeds_for_80pct_power']} paired seeds are needed for 80% power at the observed effect. The much smaller protein-level Wilcoxon p-value ({paper['protein_level_wilcoxon_p']:.3g}) is descriptive because proteins are repeated outcomes inside each seed.

The architecture comparison is also capacity-confounded: totalVI used width 256, FusionVI branches used width 128, and FusionVI reused its RNA branch for library size rather than retaining totalVI's second encoder. These implementation choices explain most of the {parameter_reduction:.1f}% parameter difference. Same-width, parameter-matched and missing-panel-aware joint-encoder controls are therefore part of the confirmatory benchmark. More decisively, the simple RNA baseline outperformed both neural models on the primary endpoint, so the 0.006 neural-model gap is not evidence of a practically better imputation method. Paired modality-dropout arms now test whether training the decoder on protein-supervised RNA-only latents closes that gap.

![Paper-aligned complete-panel benchmark](results/figures/paper_benchmark_totalvi_vs_fusionvi.png)

## Combined interpretation

The four experiments do not support a blanket claim that FusionVI is always superior. They support three narrower conclusions:

1. The complete missing-panel benchmark suggests a small FusionVI-versus-totalVI difference, but four seeds and unequal model capacity do not identify gated fusion as its cause; an RNA SVD-ridge baseline performs substantially better than both.
2. For targeted cross-mouse marker recovery, measured protein context is more important than either learned latent representation.
3. Multimodal context improves prediction of unseen PD-L1 perturbation effects, but post-transcriptional mechanisms such as CMTM6 remain difficult when RNA and surface protein move in opposite directions.

These results are relevant to therapeutic discovery as a biomarker-completion and perturbation-ranking study. They do not establish clinical utility, patient-level generalization or replacement of prospective protein measurements.

## Reproducibility and limitations

- Experiment 1 uses 10 independent donors but one dose and one 24-hour timepoint.
- Experiment 2 uses one IFN-gamma-treated cell line; it tests mechanism-level transfer rather than patient response.
- Experiment 3 has only two mice and should be treated as a technical replication.
- Experiment 4 covers one source-target batch pair and four seeds rather than the paper's 30 initializations.
- Experiment 4 seed-level inference is primary; per-protein tests are descriptive repeated-outcome summaries.
- The original Experiment 4 comparison differs in encoder width and library-network design. `run_controls.ps1` executes the prespecified capacity and missingness controls.
- Native-decoder and cross-modal-readout results are never pooled because they measure different contributions.
- The RNA-only baseline, modality-dropout arms and biological secondary metrics are prespecified in code; modality-dropout full runs and neural-model foreground/cell-type metrics remain pending.
- `results/fusionvi_experiments_summary.json` records the consolidated metrics used in this report.
- `run_paper_benchmark.ps1` reproduces the current full-panel benchmark. The original Lawlor and Papalexi training pipelines were not retained, so Experiments 1 and 2 are documented from their saved held-out predictions, metrics and figures and are not reproducible from a fresh clone. The versioned `papalexi_effects.csv` does reproduce the target-cluster bootstrap with `stats_heldout_experiments.py exp2`.

## References

1. Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nature Methods*. 2021;18:272–282. https://doi.org/10.1038/s41592-020-01050-x
2. Lawlor N, et al. Multiomic profiling identifies transcriptional and protein-level immune responses to stimulation. *Frontiers in Immunology*. 2021.
3. Papalexi E, et al. Mapping and analysis of perturbation responses in single cells by pooled CRISPR screening. *Nature Genetics*. 2021.
"""
    OUT.write_text(text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
