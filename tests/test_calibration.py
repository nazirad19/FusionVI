"""Fast checks for the calibration re-evaluation (no data or trained model needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_calibration import affine_calibrate, protein_metrics  # noqa: E402
from fusionvi import FusionVIEncoder  # noqa: E402


class TestCalibration(unittest.TestCase):
    def test_rmsle_decomposes_into_bias_and_residual(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.gamma(2.0, 1.0, size=(500, 4))
        pred = truth + 0.7 + rng.normal(0, 0.3, size=truth.shape)
        m = protein_metrics(pred, truth)
        np.testing.assert_allclose(m["rmsle"] ** 2, m["bias"] ** 2 + m["residual_sd"] ** 2, rtol=1e-10)
        self.assertTrue(np.all(np.abs(m["bias"] - 0.7) < 0.05))

    def test_affine_calibration_removes_known_offset_and_scale(self) -> None:
        rng = np.random.default_rng(1)
        x_fit = rng.uniform(0, 5, size=(1000, 3))
        y_fit = 0.5 * x_fit - 0.2
        x_apply = rng.uniform(1, 5, size=(200, 3))
        np.testing.assert_allclose(affine_calibrate(x_fit, y_fit, x_apply), 0.5 * x_apply - 0.2, atol=1e-8)

    def test_calibration_does_not_change_rank_metrics(self) -> None:
        rng = np.random.default_rng(2)
        truth = 3.0 + rng.gamma(2.0, 1.0, size=(400, 2))   # far from 0, so the >=0 clip never binds
        raw = 2.0 + 1.3 * truth + rng.normal(0, 0.5, size=truth.shape)
        cal = affine_calibrate(raw, truth, raw)
        before, after = protein_metrics(raw, truth), protein_metrics(cal, truth)
        np.testing.assert_allclose(before["pearson"], after["pearson"], atol=1e-6)
        self.assertLess(after["rmsle"].mean(), before["rmsle"].mean())

    def test_hidden_route_forces_rna_only_gate_on_source_cells(self) -> None:
        encoder = FusionVIEncoder(
            n_genes=3, n_proteins=2, n_latent=2, masked_protein_indices=[], n_cat_list=[2],
            n_layers=1, n_hidden=8, dropout_rate=0.0, available_batch_indices=[0],
        ).eval()
        data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        source_batch = torch.tensor([[0]])
        with torch.no_grad():
            _, _, gate_seen = encoder.encode_branches(data, source_batch)
            encoder.available_batch_indices = ()          # what evaluate_calibration.predict(hide_protein=True) does
            _, _, gate_hidden = encoder.encode_branches(data, source_batch)
        self.assertLess(float(gate_seen), 1.0)
        self.assertEqual(float(gate_hidden), 1.0)


if __name__ == "__main__":
    unittest.main()
