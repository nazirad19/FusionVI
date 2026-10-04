"""Leakage and design checks for the development and partial-panel benchmark objects.

Run after src/prepare_benchmarks.py:  python -m unittest tests.test_benchmarks
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import data_path, eval_proteins, load_config  # noqa: E402


def load(name: str):
    cfg = load_config(name)
    return cfg, sc.read_h5ad(data_path(cfg))


class TestBenchmarks(unittest.TestCase):
    def test_frozen_modality_dropout_sweep(self) -> None:
        cfg = load_config("dev_random")
        expected = {
            "totalvi_moddrop_p25": 0.25,
            "totalvi_moddrop_p50": 0.50,
            "totalvi_moddrop_p75": 0.75,
        }
        for name, rate in expected.items():
            arm = cfg["arms"][name]
            self.assertEqual(arm["encoder"], "joint_moddrop")
            self.assertEqual(int(arm["hidden"]), 256)
            self.assertAlmostEqual(float(arm["modality_dropout"]), rate)

    def test_hidden_target_values_never_enter_training_matrix(self) -> None:
        for name in ("dev_random", "dev_tissue", "sln206_partial", "dev_sln206_partial"):
            cfg, adata = load(name)
            target = (adata.obs["batch"].astype(str) == cfg["target_batch"]).to_numpy()
            hidden = eval_proteins(adata)
            counts = adata.obsm["protein_counts"].loc[target, hidden].to_numpy()
            truth = adata.obsm["protein_truth"].loc[target, hidden].to_numpy()
            self.assertEqual(float(counts.sum()), 0.0, name)
            self.assertGreater(float(truth.sum()), 0.0, name)

    def test_development_objects_exclude_the_real_test_mouse(self) -> None:
        for name, real_target in (("dev_random", "SLN111-D2"), ("dev_tissue", "SLN111-D2"),
                                  ("dev_sln206_partial", "SLN206-D2")):
            _, adata = load(name)
            self.assertNotIn(real_target, set(adata.obs["batch"].astype(str)), name)
            self.assertNotIn("mouse1", set(adata.obs["mouse"].astype(str)), name)

    def test_partial_panel_keeps_the_sln111_panel_observed(self) -> None:
        cfg, adata = load("sln206_partial")
        target = (adata.obs["batch"].astype(str) == cfg["target_batch"]).to_numpy()
        hidden = set(eval_proteins(adata))
        observed = [p for p in adata.obsm["protein_truth"].columns if p not in hidden]
        self.assertEqual(len(observed), 110)
        self.assertEqual(len(hidden), 97)
        self.assertGreater(float(adata.obsm["protein_counts"].loc[target, observed].to_numpy().sum()), 0.0)
        self.assertIn(cfg["target_batch"], cfg["panel_available_batches"])

    def test_complete_missing_designs_route_target_rna_only(self) -> None:
        for name in ("paper", "dev_random", "dev_tissue"):
            cfg = load_config(name)
            self.assertNotIn(cfg["target_batch"], cfg["panel_available_batches"], name)


if __name__ == "__main__":
    unittest.main()
