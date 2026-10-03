"""Leakage-safe statistics for Experiments 1 (Lawlor) and 2 (Papalexi), plus Holm.

The original Lawlor and Papalexi training pipelines are not in this repository.
The saved Papalexi held-out effects are versioned at
``results/papalexi_effects.csv``; a Lawlor cell table must be supplied in the
layout below. Run:

  python src/stats_heldout_experiments.py exp2 --csv results/papalexi_effects.csv \
      --candidate fusionvi_xmodal --reference totalvi_xmodal
  python src/stats_heldout_experiments.py exp1 --csv lawlor_cells.csv
  python src/stats_heldout_experiments.py holm --p exp1=0.31 exp2=0.04 exp3=0.50 exp4=0.20

----------------------------------------------------------------------------
exp2 input: one row per target x replicate effect (75 rows)
  target, replicate, observed            observed PD-L1 effect (e.g. log fold change vs NT)
  pred_<model> ...                       one column per model, same units
Unit of resampling = CRISPR target (its replicates move together).

exp1 input: one row per held-out cell per marker (test-fold predictions only)
  donor, marker, observed, rna           observed = log1p held-out protein;
                                         rna = log1p matching transcript
  pred_<model> ...                       one column per model
Strata and discordance thresholds for donor d are computed from all OTHER
donors only, so the held-out protein never defines its own evaluation subset.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

rng = np.random.default_rng(0)


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    out, running = {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, (len(keys) - rank) * p[k])
        out[k] = min(1.0, running)
    return out


# --------------------------------------------------------------------- exp2
def exp2_metrics(df: pd.DataFrame, model: str) -> dict[str, float]:
    pred, obs = df[f"pred_{model}"], df["observed"]
    return {
        "effect_spearman": rho(obs, pred),
        "direction_accuracy": float((np.sign(pred) == np.sign(obs)).mean()),
        "effect_mae": float(np.abs(pred - obs).mean()),
    }


def run_exp2(args) -> None:
    df = pd.read_csv(args.csv)
    models = [c[5:] for c in df.columns if c.startswith("pred_")]
    targets = df["target"].unique()
    groups = {t: df[df["target"] == t] for t in targets}

    point = {m: exp2_metrics(df, m) for m in models}
    boots = {m: [] for m in models}
    diffs = []
    for _ in range(args.n_boot):
        sample = pd.concat([groups[t] for t in rng.choice(targets, targets.size, replace=True)])
        stats = {m: exp2_metrics(sample, m) for m in models}
        for m in models:
            boots[m].append(stats[m])
        diffs.append({k: stats[args.candidate][k] - stats[args.reference][k] for k in stats[args.candidate]})

    def ci(values):
        lo, hi = np.nanpercentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    report = {"unit": f"CRISPR target (n={targets.size})", "models": {}, "contrast": {}}
    for m in models:
        report["models"][m] = {k: {"estimate": round(v, 4), "ci95": ci([b[k] for b in boots[m]])} for k, v in point[m].items()}
    for k in point[args.candidate]:
        dist = np.array([d[k] for d in diffs])
        report["contrast"][k] = {
            "difference": round(point[args.candidate][k] - point[args.reference][k], 4),
            "ci95": ci(dist),
            # two-sided bootstrap p: how often the resampled difference crosses zero
            "p_boot": round(float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))), 4),
        }
    # Sensitivity: average replicates first, then correlate over 25 independent targets.
    per_target = df.groupby("target")[["observed"] + [f"pred_{m}" for m in models]].mean()
    report["replicate_averaged_spearman_n_targets"] = {m: round(rho(per_target["observed"], per_target[f"pred_{m}"]), 4) for m in models}
    # Per-target absolute error, paired over targets (for the Wilcoxon you already report).
    err = df.assign(**{f"err_{m}": (df[f"pred_{m}"] - df["observed"]).abs() for m in models}).groupby("target").mean(numeric_only=True)
    worst = (err[f"err_{args.candidate}"]).sort_values(ascending=False).head(5)
    report["largest_candidate_errors"] = worst.round(4).to_dict()
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------- exp1
def run_exp1(args) -> None:
    df = pd.read_csv(args.csv)
    models = [c[5:] for c in df.columns if c.startswith("pred_")]
    rows = []
    for (marker, donor), test in df.groupby(["marker", "donor"]):
        train = df[(df["marker"] == marker) & (df["donor"] != donor)]

        # (a) RNA strata: cut points from other donors' RNA only.
        positive = train["rna"][train["rna"] > 0]
        cuts = np.quantile(positive, [1 / 3, 2 / 3]) if positive.size else [np.inf, np.inf]
        stratum = np.select(
            [test["rna"] == 0, test["rna"] <= cuts[0], test["rna"] <= cuts[1]],
            ["rna_zero", "rna_low", "rna_mid"], default="rna_high",
        )
        for s in ("rna_zero", "rna_low", "rna_mid", "rna_high"):
            sub = test[stratum == s]
            for m in models:
                rows.append({"marker": marker, "donor": donor, "subset": s, "model": m,
                             "n": len(sub), "spearman": rho(sub["observed"], sub[f"pred_{m}"])})

        # (b) Discordance with thresholds fixed on other donors + matched null.
        coef = np.polyfit(train["rna"], train["observed"], 1)
        thr = np.quantile(np.abs(train["observed"] - np.polyval(coef, train["rna"])), 1 - args.discordant_frac)
        resid = np.abs(test["observed"] - np.polyval(coef, test["rna"]))
        disc = test[resid > thr]
        if len(disc) >= 10:
            bins = pd.qcut(test["observed"].rank(method="first"), 10, labels=False)
            want = bins[disc.index].value_counts()
            for m in models:
                observed_rho = rho(disc["observed"], disc[f"pred_{m}"])
                null = []
                for _ in range(args.n_null):
                    idx = np.concatenate([rng.choice(bins[bins == b].index, n, replace=False) for b, n in want.items()])
                    null.append(rho(test.loc[idx, "observed"], test.loc[idx, f"pred_{m}"]))
                rows.append({"marker": marker, "donor": donor, "subset": "discordant", "model": m, "n": len(disc),
                             "spearman": observed_rho, "null_mean": float(np.nanmean(null)),
                             "null_percentile": float((np.array(null) < observed_rho).mean())})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    summary = out.groupby(["marker", "subset", "model"]).agg(
        mean_spearman=("spearman", "mean"), donors=("donor", "nunique"), median_n=("n", "median"),
        **({"null_mean": ("null_mean", "mean")} if "null_mean" in out else {}),
    ).round(3)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary)
    print(f"\nPer-donor rows written to {args.out}. Donor is the unit: compare models with a paired test over donors.")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("exp2")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--candidate", default="fusionvi_xmodal")
    p2.add_argument("--reference", default="totalvi_xmodal")
    p2.add_argument("--n-boot", type=int, default=5000)
    p1 = sub.add_parser("exp1")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--discordant-frac", type=float, default=0.10)
    p1.add_argument("--n-null", type=int, default=500)
    p1.add_argument("--out", default="results/lawlor_strata_metrics.csv")
    ph = sub.add_parser("holm")
    ph.add_argument("--p", nargs="+", required=True, help="name=pvalue pairs, one per primary endpoint")
    args = parser.parse_args()
    if args.cmd == "exp2":
        run_exp2(args)
    elif args.cmd == "exp1":
        run_exp1(args)
    else:
        raw = {k: float(v) for k, v in (item.split("=") for item in args.p)}
        print(json.dumps({k: {"p": raw[k], "p_holm": round(v, 4)} for k, v in holm(raw).items()}, indent=2))


if __name__ == "__main__":
    main()
