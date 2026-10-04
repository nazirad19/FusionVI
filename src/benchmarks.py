"""Benchmark registry: resolves data paths, batches and output folders by name.

`paper` keeps the original folders (results/paper_benchmark_runs,
models/paper_benchmark) so existing runs are reused unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "paper_benchmark.yaml"


def load_config(benchmark: str = "paper") -> dict:
    """Full config with source/target/availability overridden for `benchmark`."""
    cfg = yaml.safe_load(CONFIG.read_text())
    if benchmark not in cfg["benchmarks"]:
        raise SystemExit(f"Unknown benchmark {benchmark!r}; choose from {sorted(cfg['benchmarks'])}")
    bench = cfg["benchmarks"][benchmark]
    out = copy.deepcopy(cfg)
    out["benchmark_name"] = benchmark
    out["source_batch"] = bench["source_batch"]
    out["target_batch"] = bench["target_batch"]
    out["panel_available_batches"] = list(bench["panel_available_batches"])
    out["stage"] = bench.get("stage", "final")
    return out


def data_path(cfg: dict) -> Path:
    return ROOT / "data" / "processed" / cfg["benchmarks"][cfg["benchmark_name"]]["data"]


def runs_dir(cfg: dict) -> Path:
    name = cfg["benchmark_name"]
    return ROOT / "results" / ("paper_benchmark_runs" if name == "paper" else f"runs_{name}")


def models_dir(cfg: dict) -> Path:
    name = cfg["benchmark_name"]
    return ROOT / "models" / ("paper_benchmark" if name == "paper" else name)


def output_suffix(cfg: dict) -> str:
    return "" if cfg["benchmark_name"] == "paper" else f"_{cfg['benchmark_name']}"


def eval_proteins(adata) -> list[str]:
    """Proteins scored on the target batch (all, unless the object lists a hidden subset)."""
    names = list(map(str, adata.obsm["protein_truth"].columns))
    listed = adata.uns.get("eval_proteins")
    return names if listed is None else [p for p in names if p in set(map(str, listed))]
