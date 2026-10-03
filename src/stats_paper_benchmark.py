"""Seed-level inference for the paper-aligned missing-protein benchmark (Experiment 4).

The random initialization is the unit of replication. Proteins are repeated
measurements within a seed, so protein-level tests are reported only as
descriptive summaries.

Usage:
    python src/stats_paper_benchmark.py [--metrics results/paper_benchmark_all_metrics.csv]
                                        [--reference totalVI] [--candidate FusionVI]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["rmsle", "mae_log1p", "mae_raw", "spearman", "pearson_log1p"]
LOWER_IS_BETTER = {"rmsle": True, "mae_log1p": True, "mae_raw": True, "spearman": False, "pearson_log1p": False}


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation p-value on the mean."""
    observed = abs(d.mean())
    flips = np.array(list(itertools.product([-1, 1], repeat=d.size)))
    null = np.abs((flips * d).mean(axis=1))
    return float((null >= observed - 1e-15).mean())


def bootstrap_ci(d: np.ndarray, rng: np.random.Generator, n: int = 20_000) -> tuple[float, float]:
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def seeds_needed(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired-t sample size for detecting a mean difference delta with seed SD sd."""
    if sd == 0 or delta == 0:
        return 0
    for n in range(3, 1000):
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ncp = abs(delta) / (sd / np.sqrt(n))
        achieved = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        if achieved >= power:
            return n
    return 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "results" / "paper_benchmark_all_metrics.csv"))
    parser.add_argument("--reference", default="totalVI")
    parser.add_argument("--candidate", default="FusionVI")
    parser.add_argument("--out", default=str(ROOT / "results" / "paper_benchmark_seed_inference.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    rng = np.random.default_rng(0)
    out: dict = {"reference": args.reference, "candidate": args.candidate, "unit": "random initialization (seed)", "metrics": {}}

    for metric in METRICS:
        wide = df.pivot_table(index=["seed", "protein"], columns="model", values=metric)
        wide = wide.dropna(subset=[args.reference, args.candidate])
        per_seed = wide.groupby("seed")[[args.reference, args.candidate]].mean()
        d = (per_seed[args.candidate] - per_seed[args.reference]).to_numpy()
        n = d.size
        sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        t_res = stats.ttest_1samp(d, 0.0) if n > 1 else None
        t_half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
        favourable = (d < 0) if LOWER_IS_BETTER[metric] else (d > 0)

        # Descriptive only: protein-level paired differences averaged over seeds.
        prot = (wide[args.candidate] - wide[args.reference]).groupby(level="protein").mean()
        prot_fav = (prot < 0) if LOWER_IS_BETTER[metric] else (prot > 0)

        out["metrics"][metric] = {
            "lower_is_better": LOWER_IS_BETTER[metric],
            "reference_mean": float(per_seed[args.reference].mean()),
            "candidate_mean": float(per_seed[args.candidate].mean()),
            "per_seed_difference": {int(s): float(v) for s, v in zip(per_seed.index, d)},
            "mean_difference": float(d.mean()),
            "relative_difference_pct": float(100 * d.mean() / per_seed[args.reference].mean()),
            "seed_sd_of_difference": sd,
            "t_ci95": [float(d.mean() - t_half), float(d.mean() + t_half)],
            "bootstrap_ci95": list(bootstrap_ci(d, rng)) if n > 1 else None,
            "paired_t_p": float(t_res.pvalue) if t_res is not None else None,
            "sign_flip_exact_p": sign_flip_p(d) if n <= 20 else None,
            "min_attainable_sign_flip_p": float(2 / 2**n),
            "seeds_favouring_candidate": int(favourable.sum()),
            "n_seeds": int(n),
            "seeds_for_80pct_power_at_observed_effect": seeds_needed(sd, float(d.mean())) if n > 1 else None,
            "descriptive_proteins_favouring_candidate": int(prot_fav.sum()),
            "descriptive_n_proteins": int(prot.size),
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    rows = []
    for metric, r in out["metrics"].items():
        rows.append({
            "metric": metric,
            args.reference: round(r["reference_mean"], 4),
            args.candidate: round(r["candidate_mean"], 4),
            "diff": round(r["mean_difference"], 4),
            "t 95% CI": f"[{r['t_ci95'][0]:.4f}, {r['t_ci95'][1]:.4f}]",
            "p (t)": round(r["paired_t_p"], 3),
            "p (exact)": r["sign_flip_exact_p"],
            "seeds fav.": f"{r['seeds_favouring_candidate']}/{r['n_seeds']}",
            "seeds for 80% power": r["seeds_for_80pct_power_at_observed_effect"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
