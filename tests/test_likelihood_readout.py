"""Tests for the likelihood-consistent totalVI protein readout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scvi  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from scvi.model import TOTALVI  # noqa: E402

from benchmark_arms import build_model  # noqa: E402
from compare_methods_calibrated import neural_readouts  # noqa: E402
from evaluate_calibration import _mixture_mean, predict, protein_metrics, source_efficiency  # noqa: E402

N_CELLS, N_GENES, N_PROTEINS = 80, 30, 6
EFFICIENCY = np.array([0.37, 0.5, 0.8, 1.3, 2.0, 0.25], dtype=np.float32)


def make_model():
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    proteins = rng.poisson(20.0, size=(N_CELLS, N_PROTEINS)).astype(np.float32)
    batch = np.where(np.arange(N_CELLS) < N_CELLS // 2, "SRC", "TGT")
    proteins[batch == "TGT"] = 0.0
    adata = ad.AnnData(X=counts)
    adata.obs_names = [f"c{i}" for i in range(N_CELLS)]
    adata.layers["counts"] = counts.copy()
    adata.obs["batch"] = pd.Categorical(batch, categories=["SRC", "TGT"])
    adata.obsm["protein_counts"] = pd.DataFrame(
        proteins,
        index=adata.obs_names,
        columns=[f"p{j}" for j in range(N_PROTEINS)],
    )
    TOTALVI.setup_anndata(
        adata,
        batch_key="batch",
        layer="counts",
        protein_expression_obsm_key="protein_counts",
    )
    cfg = {
        "n_latent": 4,
        "totalvi_hidden": 16,
        "source_batch": "SRC",
        "panel_available_batches": ["SRC"],
    }
    model = build_model(adata, cfg, {"encoder": "joint", "hidden": 16})
    with torch.no_grad():
        model.module.log_per_batch_efficiency[:, 0] = torch.log(torch.tensor(EFFICIENCY))
    model.module.eval()
    return model, adata


class TestLikelihoodReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scvi.settings.verbosity = 0
        cls.model, cls.adata = make_model()
        cls.target = np.where(cls.adata.obs["batch"].astype(str).to_numpy() == "TGT")[0]

    def readouts(self, seed: int = 7):
        torch.manual_seed(seed)
        return predict(
            self.model,
            self.adata,
            self.target,
            source_code=0,
            n_samples=5,
            batch_size=len(self.target),
        )

    def test_matches_direct_py_norm_computation(self) -> None:
        out = self.readouts()
        torch.manual_seed(7)
        tensors = next(iter(self.model._make_data_loader(
            adata=self.adata,
            indices=self.target,
            batch_size=len(self.target),
            shuffle=False,
        )))
        with torch.no_grad():
            _, generated = self.model.module.forward(
                tensors,
                inference_kwargs={"n_samples": 5},
                generative_kwargs={"transform_batch": 0},
                compute_loss=False,
            )
        direct = torch.log1p(_mixture_mean(generated["py_norm_"])).numpy()
        np.testing.assert_allclose(out["likelihood_mean"], direct, rtol=1e-5, atol=1e-6)

    def test_equals_helper_rescaled_by_source_efficiency(self) -> None:
        out = self.readouts()
        np.testing.assert_allclose(
            np.expm1(out["likelihood_mean"]),
            np.expm1(out["helper_mean"]) * EFFICIENCY,
            rtol=1e-4,
        )

    def test_reports_source_batch_efficiency(self) -> None:
        measured = source_efficiency(self.model, 0)
        np.testing.assert_allclose(measured, EFFICIENCY, rtol=1e-6, atol=1e-6)

    def test_correction_changes_rmsle_but_not_spearman(self) -> None:
        out = self.readouts()
        noise = np.random.default_rng(1).gamma(5, 0.2, out["likelihood_mean"].shape)
        truth = np.log1p(np.expm1(out["likelihood_mean"]) * noise)
        helper = protein_metrics(out["helper_mean"], truth)
        likelihood = protein_metrics(out["likelihood_mean"], truth)
        self.assertGreater(float(np.abs(helper["rmsle"] - likelihood["rmsle"]).max()), 1e-3)
        for j in range(N_PROTEINS):
            self.assertAlmostEqual(
                spearmanr(truth[:, j], out["helper_mean"][:, j]).statistic,
                spearmanr(truth[:, j], out["likelihood_mean"][:, j]).statistic,
                places=10,
            )

    def test_comparison_script_agrees_with_evaluator(self) -> None:
        out = self.readouts(seed=11)
        torch.manual_seed(11)
        other = neural_readouts(
            self.model,
            self.adata,
            self.target,
            source_code=0,
            n_samples=5,
            hide_protein=False,
            hide_columns=None,
            batch_size=len(self.target),
        )
        np.testing.assert_allclose(other["likelihood_mean"], out["likelihood_mean"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(other["helper_mean"], out["helper_mean"], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
