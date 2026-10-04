# FusionVI

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
| FusionVI | 0.5678 |
| Published totalVI | 0.5650 |
| Joint encoder, width 128 | 0.5660 |
| Joint encoder, parameter matched | 0.5698 |
| Joint encoder plus availability | 0.5662 |

FusionVI-minus-reference paired RMSLE differences:

- published totalVI: +0.0027
  (95% CI +0.0020 to +0.0035);
- same-width totalVI: +0.0018
  (95% CI +0.0008 to +0.0027);
- parameter-matched totalVI: -0.0021
  (95% CI -0.0030 to -0.0011);
- availability-aware totalVI: +0.0015
  (95% CI +0.0008 to +0.0023).

The availability-indicator-minus-same-width contrast is
+0.0002 RMSLE
(95% CI -0.0005 to +0.0010).
Spearman and within-cell-type Spearman are interpreted separately because a
per-protein efficiency factor cannot change within-protein ranks.

## Reproduce the re-evaluation

```powershell
python src\evaluate_calibration.py --benchmark paper --arms totalvi fusionvi totalvi_w128 totalvi_pmatch totalvi_avail_w128 --tag likelihood_correct
python src\plot_calibration.py --tag likelihood_correct
python -m unittest discover -s tests -v
```

The run reloads saved checkpoints and performs inference only. No retraining or
target-protein fitting occurs.

## Reference

Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell
multi-omic data with totalVI. *Nature Methods*. 2021;18:272–282.
https://doi.org/10.1038/s41592-020-01050-x
