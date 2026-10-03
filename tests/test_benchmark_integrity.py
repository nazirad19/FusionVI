"""Fast integrity checks for the paper benchmark and new encoder routes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scanpy as sc
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusionvi import FusionVIEncoder  # noqa: E402


class TestBenchmarkIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adata = sc.read_h5ad(ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad")

    def test_target_protein_panel_is_zero_in_training_object(self) -> None:
        target = self.adata.obs["batch"].astype(str).to_numpy() == "SLN111-D2"
        panel = self.adata.obsm["protein_counts"].to_numpy()[target]
        self.assertEqual(float(panel.sum()), 0.0)
        self.assertGreater(float(self.adata.obsm["protein_truth"].to_numpy()[target].sum()), 0.0)

    def test_availability_uses_batch_metadata(self) -> None:
        encoder = FusionVIEncoder(
            n_genes=3,
            n_proteins=2,
            n_latent=2,
            masked_protein_indices=[],
            n_cat_list=[2],
            n_layers=1,
            n_hidden=8,
            dropout_rate=0.0,
            available_batch_indices=[0],
        ).eval()
        # First cell is a genuinely all-zero measured panel in source batch 0;
        # second cell has nonzero placeholder counts but is unavailable batch 1.
        data = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [1.0, 2.0, 3.0, 9.0, 9.0]])
        batch = torch.tensor([[0], [1]])
        with torch.no_grad():
            _, _, gate = encoder.encode_branches(data, batch)
        self.assertLess(float(gate[0]), 1.0)
        self.assertEqual(float(gate[1]), 1.0)

    def test_papalexi_effect_table_has_complete_heldout_units(self) -> None:
        import pandas as pd

        effects = pd.read_csv(ROOT / "results" / "papalexi_effects.csv")
        self.assertEqual(effects["target"].nunique(), 25)
        self.assertEqual(len(effects), 75)
        self.assertEqual(set(effects.groupby("target")["replicate"].nunique()), {3})


if __name__ == "__main__":
    unittest.main()
