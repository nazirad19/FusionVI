"""Write the public README from likelihood-consistent benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAG = "likelihood_correct"


def main() -> None:
    summary_path = ROOT / "results" / f"calibration_summary_{TAG}.csv"
    contrast_path = ROOT / "results" / f"calibration_contrasts_{TAG}.csv"
    if not summary_path.exists() or not contrast_path.exists():
        raise FileNotFoundError("Run the likelihood-consistent evaluator before regenerating README.md")
    summary = pd.read_csv(summary_path).set_index(["arm", "readout"])
    contrasts = pd.read_csv(contrast_path)
    required = {"likelihood_mean", "likelihood_pred_log", "calibrated", "helper_mean"}
    missing = required - set(summary.index.get_level_values("readout"))
    if missing:
        raise ValueError(f"Missing likelihood-consistent readouts: {sorted(missing)}")

    def value(arm: str, metric: str, readout: str = "likelihood_mean") -> float:
        return float(summary.loc[(arm, readout), metric])

    def contrast(candidate: str, reference: str, metric: str = "rmsle") -> dict:
        rows = contrasts[
            (contrasts["readout"] == "likelihood_mean")
            & (contrasts["metric"] == metric)
            & (contrasts["candidate"] == candidate)
            & (contrasts["reference"] == reference)
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one contrast for {candidate} vs {reference}, {metric}; found {len(rows)}")
        return rows.iloc[0].to_dict()

    fv_tv = contrast("fusionvi", "totalvi")
    fv_width = contrast("fusionvi", "totalvi_w128")
    fv_pmatch = contrast("fusionvi", "totalvi_pmatch")
    fv_avail = contrast("fusionvi", "totalvi_avail_w128")
    availability = contrast("totalvi_avail_w128", "totalvi_w128")

    text = f"""# FusionVI

FusionVI is an experimental totalVI encoder evaluated on missing-protein
benchmarks. It uses separate RNA and protein branches with a learned cell-level
gate while retaining the published totalVI decoder and likelihoods.

## Primary paper-aligned benchmark

The SLN111-D1 batch supplies RNA and 110 proteins. All 110 D2 proteins are
hidden during training and used only for scoring. Sixteen paired random seeds
were evaluated from saved 500-epoch checkpoints.

The primary readout is the likelihood-consistent expected protein count from
`py_norm_`, including each checkpoint's learned per-protein efficiency.
`get_normalized_expression()` returns a normalized representation from `py_`
that omits this factor and is retained only as a historical audit readout.

| Encoder | Mean RMSLE |
|---|---:|
| FusionVI | {value('fusionvi', 'rmsle'):.4f} |
| Published totalVI | {value('totalvi', 'rmsle'):.4f} |
| Joint encoder, width 128 | {value('totalvi_w128', 'rmsle'):.4f} |
| Joint encoder, parameter matched | {value('totalvi_pmatch', 'rmsle'):.4f} |
| Joint encoder plus availability | {value('totalvi_avail_w128', 'rmsle'):.4f} |

FusionVI-minus-reference paired RMSLE differences:

- published totalVI: {fv_tv['candidate_minus_ref']:+.4f}
  (95% CI {fv_tv['ci95_low']:+.4f} to {fv_tv['ci95_high']:+.4f});
- same-width totalVI: {fv_width['candidate_minus_ref']:+.4f}
  (95% CI {fv_width['ci95_low']:+.4f} to {fv_width['ci95_high']:+.4f});
- parameter-matched totalVI: {fv_pmatch['candidate_minus_ref']:+.4f}
  (95% CI {fv_pmatch['ci95_low']:+.4f} to {fv_pmatch['ci95_high']:+.4f});
- availability-aware totalVI: {fv_avail['candidate_minus_ref']:+.4f}
  (95% CI {fv_avail['ci95_low']:+.4f} to {fv_avail['ci95_high']:+.4f}).

The availability-indicator-minus-same-width contrast is
{availability['candidate_minus_ref']:+.4f} RMSLE
(95% CI {availability['ci95_low']:+.4f} to {availability['ci95_high']:+.4f}).
Spearman and within-cell-type Spearman are interpreted separately because a
per-protein efficiency factor cannot change within-protein ranks.

## Reproduce the re-evaluation

```powershell
python src\\evaluate_calibration.py --benchmark paper --arms totalvi fusionvi totalvi_w128 totalvi_pmatch totalvi_avail_w128 --tag likelihood_correct
python src\\plot_calibration.py --tag likelihood_correct
python -m unittest discover -s tests -v
```

The run reloads saved checkpoints and performs inference only. No retraining or
target-protein fitting occurs.

## Reference

Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell
multi-omic data with totalVI. *Nature Methods*. 2021;18:272–282.
https://doi.org/10.1038/s41592-020-01050-x
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print(ROOT / "README.md")


if __name__ == "__main__":
    main()
