"""Write the consolidated Markdown report for all completed FusionVI experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "fusionvi_experiments_summary.json"
OUT = ROOT / "TECHNICAL_REPORT.md"


def main() -> None:
    lawlor, papalexi, targeted, paper = json.loads(SUMMARY.read_text())["experiments"]
    pm = papalexi["metrics"]
    pb = papalexi["target_cluster_bootstrap"]
    pt = papalexi["paired_target_test"]
    ridge = paper["rna_ridge_baseline"]
    controls = paper["confirmatory_controls"]
    cmeans = controls["mean_rmsle"]
    contrasts = {(row["candidate"], row["reference"]): row for row in controls["contrasts"]}
    same_width = contrasts[("FusionVI", "totalvi_w128")]
    param_match = contrasts[("FusionVI", "totalvi_pmatch")]
    availability = contrasts[("FusionVI", "totalvi_avail_w128")]
    indicator_only = contrasts[("totalvi_avail_w128", "totalvi_w128")]
    original_16 = contrasts[("FusionVI", "totalVI")]
    calibration = pd.read_csv(ROOT / "results" / "calibration_summary.csv").set_index(["arm", "readout"])
    calibration_contrasts = pd.read_csv(ROOT / "results" / "calibration_contrasts.csv")

    def cal(arm: str, readout: str, metric: str) -> float:
        return float(calibration.loc[(arm, readout), metric])

    def cal_contrast(metric: str, reference: str = "totalvi") -> dict:
        row = calibration_contrasts[
            (calibration_contrasts["readout"] == "calibrated")
            & (calibration_contrasts["metric"] == metric)
            & (calibration_contrasts["reference"] == reference)
        ].iloc[0]
        return row.to_dict()

    calibrated_rmsle = cal_contrast("rmsle")
    calibrated_residual = cal_contrast("residual_sd")
    calibrated_within = cal_contrast("within_celltype_spearman")
    relative = 100 * (paper["totalvi_mean_rmsle"] - paper["fusionvi_mean_rmsle"]) / paper["totalvi_mean_rmsle"]
    parameter_reduction = 100 * (paper["totalvi_parameters"] - paper["fusionvi_parameters"]) / paper["totalvi_parameters"]

    text = f"""# FusionVI technical report

## Project question

Can modality-specific fusion improve recovery of therapeutic surface biomarkers when protein measurements are missing, discordant with RNA, or observed under an unseen perturbation?

FusionVI replaces totalVI's joint encoder with separate RNA and protein branches and a cell-level gate while retaining the totalVI decoder and likelihoods. The latest missing-panel version explicitly routes cells through the RNA branch when the complete protein panel is absent. Four experiments test different forms of generalization. In the complete-panel experiment, FusionVI had a small advantage under the original log1p-of-expected-count RMSLE, but the advantage disappeared after a D1-only scale calibration. Its residual error and within-cell-type ranking were slightly worse than totalVI. The controlled result therefore identifies output calibration, rather than additional biological information, as the main source of the original RMSLE difference.

## Experiment overview

| Experiment | Dataset and held-out unit | Biological question | Main result |
|---|---|---|---|
| 1. Donor-held-out activation | Lawlor PBMC CITE-seq; 10 donors | Can hidden CD25, CD69 and HLA-DR be recovered in an unseen donor, including RNA-protein-discordant cells? | Native encoders were close; a donor-safe cross-modal readout provided most of the gain. |
| 2. Unseen perturbations | Papalexi ECCITE-seq; 25 CRISPR targets | Given a held-out perturbation cell's RNA and three measured proteins, can its hidden PD-L1 response be recovered? | FusionVI-X achieved effect Spearman {pm['fusionvi_xmodal']['effect_spearman']:.3f} and 84.0% direction accuracy. |
| 3. Targeted cross-mouse transfer | Original totalVI SLN111; 2 mice | Can four hidden immune markers be transferred to an unseen mouse? | A latent-free protein-context baseline reached {targeted['protein_context_plus_rna_mean_spearman']:.3f}; adding either latent changed Spearman by no more than {max(targeted['totalvi_latent_increment'], targeted['fusionvi_latent_increment']):.3f}. |
| 4. Complete missing panel | totalVI Figure 3 SLN111 design; 16 paired seeds | Can RNA recover all 110 proteins in a batch with no protein input? | FusionVI's raw RMSLE lead disappeared after source-only calibration; information-recovery metrics did not improve. |

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

This experiment follows the totalVI paper's Figure 3 missing-protein test. SLN111-D1 retained RNA and all 110 proteins, while every D2 protein was hidden during training and preserved only for scoring. Every neural model used the same 20-dimensional latent space, native decoder and likelihoods, optimizer, split, 500-epoch budget and 25 posterior samples. Sixteen paired seeds were run; the paper used 30 initializations.

| Model | Encoder comparison | Mean RMSLE across 16 seeds | Difference versus FusionVI |
|---|---|---:|---:|
| FusionVI | Separate branches plus learned gate | **{cmeans['FusionVI']:.4f}** | reference |
| totalVI | Published joint encoder, width 256 | {cmeans['totalVI']:.4f} | {-original_16['rmsle_diff']:+.4f} |
| Joint, same width | Width 128 | {cmeans['totalvi_w128']:.4f} | {-same_width['rmsle_diff']:+.4f} |
| Joint, parameter matched | Width 70 | {cmeans['totalvi_pmatch']:.4f} | {-param_match['rmsle_diff']:+.4f} |
| Joint plus availability | Width 128 plus missing-panel indicator | {cmeans['totalvi_avail_w128']:.4f} | {-availability['rmsle_diff']:+.4f} |

The original readout computes log1p of the expected protein count. Because the protein likelihood is overdispersed, this differs from the expected log1p count targeted by RMSLE. We therefore reloaded every checkpoint without retraining and evaluated three readouts: the original log1p(E[y]), posterior E[log1p y], and a per-protein affine calibration fitted only on D1 cells passed through the same RNA-only route as D2.

| Readout | totalVI RMSLE | FusionVI RMSLE | FusionVI − totalVI |
|---|---:|---:|---:|
| Original log1p(E[y]) | {cal('totalvi', 'log_mean', 'rmsle'):.4f} | {cal('fusionvi', 'log_mean', 'rmsle'):.4f} | {original_16['rmsle_diff']:+.4f} |
| Posterior E[log1p y] | {cal('totalvi', 'pred_log', 'rmsle'):.4f} | {cal('fusionvi', 'pred_log', 'rmsle'):.4f} | {cal('fusionvi', 'pred_log', 'rmsle')-cal('totalvi', 'pred_log', 'rmsle'):+.4f} |
| D1-only calibrated | **{cal('totalvi', 'calibrated', 'rmsle'):.4f}** | {cal('fusionvi', 'calibrated', 'rmsle'):.4f} | {calibrated_rmsle['fusionvi_minus_ref']:+.4f} |
| RNA SVD-ridge | {cal('rna_ridge', 'ridge', 'rmsle'):.4f} | — | — |

After calibration, FusionVI was numerically worse than totalVI by {calibrated_rmsle['fusionvi_minus_ref']:+.4f} RMSLE (95% CI {calibrated_rmsle['ci95_low']:+.4f} to {calibrated_rmsle['ci95_high']:+.4f}; Holm-adjusted p={calibrated_rmsle['p_holm_within_readout_metric']:.3f}). Its residual error was higher by {calibrated_residual['fusionvi_minus_ref']:+.4f} (Holm-adjusted p={calibrated_residual['p_holm_within_readout_metric']:.2g}), and its mean within-cell-type Spearman was {cal('fusionvi', 'calibrated', 'within_celltype_spearman'):.4f} versus {cal('totalvi', 'calibrated', 'within_celltype_spearman'):.4f} for totalVI. Marker AUROC was essentially saturated for both models ({cal('fusionvi', 'calibrated', 'marker_auroc'):.4f} versus {cal('totalvi', 'calibrated', 'marker_auroc'):.4f}).

The widened RNA ridge baseline reached {cal('rna_ridge', 'ridge', 'rmsle'):.4f} RMSLE. D1 calibration improved the neural models below ridge on aggregate error, confirming that the earlier ridge gap was largely a scale effect. Ridge and the neural models remained similar on overall and within-cell-type rank correlation.

Random initialization is the valid replication unit. Against the same-width joint encoder, FusionVI reduced RMSLE by {abs(same_width['rmsle_diff']):.4f} (95% CI {same_width['ci95_low']:+.4f} to {same_width['ci95_high']:+.4f}; Holm-adjusted p={same_width['p_holm']:.2g}) and won {same_width['seeds_candidate_better']}/16 seeds. Against the parameter-matched joint encoder, the reduction was {abs(param_match['rmsle_diff']):.4f} (95% CI {param_match['ci95_low']:+.4f} to {param_match['ci95_high']:+.4f}; Holm-adjusted p={param_match['p_holm']:.2g}) with 16/16 wins. Against the missingness-aware joint encoder, the reduction was {abs(availability['rmsle_diff']):.4f} (95% CI {availability['ci95_low']:+.4f} to {availability['ci95_high']:+.4f}; Holm-adjusted p={availability['p_holm']:.2g}) with {availability['seeds_candidate_better']}/16 wins. Adding the availability indicator to a same-width joint encoder did not help: difference {indicator_only['rmsle_diff']:+.4f}, 95% CI {indicator_only['ci95_low']:+.4f} to {indicator_only['ci95_high']:+.4f}, p={indicator_only['p_t']:.2f}.

The original comparison was capacity-confounded because totalVI used width 256, FusionVI used width-128 branches and FusionVI reused its RNA branch for library size. Matched controls showed that FusionVI's original-scale RMSLE difference was reproducible, but the scale-matched analysis changes its meaning: calibration removes the apparent advantage, while residual and rank metrics do not favor FusionVI. Paired modality-dropout arms are implemented but were not required for this conclusion.

![Scale-matched complete-panel evaluation](results/figures/calibration_benchmark.png)

## Combined interpretation

The four experiments do not support a blanket claim that FusionVI is always superior. They support three narrower conclusions:

1. The complete missing-panel benchmark does not support improved biological information recovery by FusionVI. Its original RMSLE lead is chiefly a calibration effect and disappears after D1-only scale matching.
2. For targeted cross-mouse marker recovery, measured protein context is more important than either learned latent representation.
3. Multimodal context improves prediction of unseen PD-L1 perturbation effects, but post-transcriptional mechanisms such as CMTM6 remain difficult when RNA and surface protein move in opposite directions.

These results are relevant to therapeutic discovery as a biomarker-completion and perturbation-ranking study. They do not establish clinical utility, patient-level generalization or replacement of prospective protein measurements.

## Reproducibility and limitations

- Experiment 1 uses 10 independent donors but one dose and one 24-hour timepoint.
- Experiment 2 uses one IFN-gamma-treated cell line; it tests mechanism-level transfer rather than patient response.
- Experiment 3 has only two mice and should be treated as a technical replication.
- Experiment 4 covers one source-target batch pair and 16 seeds rather than the paper's 30 initializations.
- Experiment 4 seed-level inference is primary; per-protein tests are descriptive repeated-outcome summaries.
- The original Experiment 4 comparison differs in encoder width and library-network design; the completed same-width, parameter-matched and missingness-aware controls address these prespecified confounders.
- Native-decoder and cross-modal-readout results are never pooled because they measure different contributions.
- Modality-dropout arms remain pending. Neural overall, within-cell-type and marker-AUROC metrics are complete for all 16 seeds and all Tier 0/1 arms.
- `results/calibration_summary.csv`, `calibration_contrasts.csv`, `calibration_within_celltype.csv` and `calibration_marker_auroc.csv` contain the scale-matched evaluation.
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
