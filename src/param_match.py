"""Find encoder widths that equalize trainable parameters across arms.

Prints, for each encoder kind, the width whose total model parameter count is
closest to (a) FusionVI at its configured width and (b) totalVI at its
configured width. Copy the results into `arms:` in paper_benchmark.yaml.
"""

from __future__ import annotations

from pathlib import Path

import scanpy as sc
import yaml
from scvi.model import TOTALVI

from benchmark_arms import build_model, n_trainable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"


def closest(adata, cfg, kind, target, lo=8, hi=1024):
    """Bisection on width (parameter count is monotone in width); even widths only."""
    count = lambda h: n_trainable(build_model(adata, cfg, {"encoder": kind, "hidden": h, "gate": "learned"}))
    while hi - lo > 2:
        mid = (lo + hi) // 2 // 2 * 2
        if count(mid) < target:
            lo = mid
        else:
            hi = mid
    candidates = [(h, count(h)) for h in (lo, hi)]
    return min(candidates, key=lambda c: abs(c[1] - target))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "paper_benchmark.yaml").read_text())
    adata = sc.read_h5ad(DATA)
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    ref_total = n_trainable(build_model(adata, cfg, {"encoder": "joint", "hidden": cfg["totalvi_hidden"]}))
    ref_fusion = n_trainable(build_model(adata, cfg, {"encoder": "fusion", "hidden": cfg["fusion_branch_hidden"]}))
    print(f"totalVI  (joint,  h={cfg['totalvi_hidden']}): {ref_total:,}")
    print(f"FusionVI (fusion, h={cfg['fusion_branch_hidden']}): {ref_fusion:,}")
    for kind in ("joint", "joint_available"):
        h, n = closest(adata, cfg, kind, ref_fusion)
        print(f"{kind:16s} matched to FusionVI: hidden={h}  params={n:,}  ({100*(n-ref_fusion)/ref_fusion:+.2f}%)")
    h, n = closest(adata, cfg, "fusion", ref_total)
    print(f"{'fusion':16s} matched to totalVI:  hidden={h}  params={n:,}  ({100*(n-ref_total)/ref_total:+.2f}%)")


if __name__ == "__main__":
    main()
