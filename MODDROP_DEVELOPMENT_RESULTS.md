# Development decision: whole-panel modality dropout

The frozen masking sweep completed 32 fits: four arms, four paired seeds and two SLN111-D1 development splits. All results use the likelihood-consistent `likelihood_mean` readout.

## Result

| Split | Arm | RMSLE | Delta RMSLE vs totalVI | RMSLE wins | Within-cell-type Spearman | Delta Spearman | Spearman wins |
|---|---|---:|---:|---:|---:|---:|---:|
| `dev_random` | totalVI | 0.5343 | — | — | 0.1630 | — | — |
| `dev_random` | dropout 0.25 | 0.5339 | −0.00047 | 2/4 | 0.1630 | +0.00005 | 2/4 |
| `dev_random` | dropout 0.50 | **0.5331** | **−0.00121** | **4/4** | 0.1659 | +0.00293 | 3/4 |
| `dev_random` | dropout 0.75 | 0.5336 | −0.00077 | 2/4 | **0.1661** | +0.00315 | 3/4 |
| `dev_tissue` | totalVI | 0.5539 | — | — | 0.1359 | — | — |
| `dev_tissue` | dropout 0.25 | 0.5529 | −0.00094 | 3/4 | 0.1379 | +0.00201 | 2/4 |
| `dev_tissue` | dropout 0.50 | **0.5512** | **−0.00266** | **4/4** | **0.1397** | +0.00377 | 3/4 |
| `dev_tissue` | dropout 0.75 | **0.5512** | −0.00265 | 3/4 | 0.1388 | +0.00290 | 2/4 |

Only dropout 0.50 passed the prespecified gate on both splits: delta RMSLE at most −0.0010 and delta within-cell-type Spearman at least −0.0020. It is therefore the selected masking rate.

The difference between 50% and 75% masking is not treated as robust. The defensible development conclusion is that moderate-to-strong masking produced a small improvement of approximately 0.001–0.003 RMSLE, with no loss of within-cell-type ranking. The 50% choice follows the frozen numerical rule rather than evidence of a stable optimum.

Median learned source-batch protein efficiencies ranged from 0.3681 to 0.3739 across all arms, seeds and splits. This is consistent with the approximately 0.37 efficiency in the paper models and provides no indication of a training-scale failure.

## Decision

`totalvi_moddrop_p50` advances to the exploratory distillation stage. If neither prespecified distillation weight passes its separate development gate, this masking-only arm remains the frozen candidate for the final cross-mouse `paper` benchmark.

Machine-readable results are in `results/moddrop_dev_gate_v1.json`. Full tagged outputs use the suffix `moddrop_dev_v1`.
