"""Encoders for the paper totalVI baseline and the final FusionVI model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from scvi.nn import EncoderTOTALVI, FCLayers


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class MaskedJointEncoderTOTALVI(nn.Module):
    """The standard totalVI encoder with selected proteins hidden at input."""

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        n_latent: int,
        masked_protein_indices: Iterable[int],
        n_cat_list: Iterable[int] | None,
        n_layers: int,
        n_hidden: int,
        dropout_rate: float,
        distribution: str = "normal",
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_proteins = int(n_proteins)
        mask = torch.ones(self.n_proteins)
        mask[list(masked_protein_indices)] = 0.0
        self.register_buffer("protein_input_mask", mask)
        self.base = EncoderTOTALVI(
            self.n_genes + self.n_proteins,
            n_latent,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=distribution,
            use_batch_norm=True,
        )
        self.z_transformation = self.base.z_transformation
        self.l_transformation = self.base.l_transformation

    def forward(self, data: torch.Tensor, *cat_list: int):
        rna = data[:, : self.n_genes]
        protein = data[:, self.n_genes : self.n_genes + self.n_proteins]
        remainder = data[:, self.n_genes + self.n_proteins :]
        protein = protein * self.protein_input_mask
        masked = torch.cat((rna, protein, remainder), dim=-1)
        return self.base(masked, *cat_list)


class FusionVIEncoder(nn.Module):
    """Separate RNA/protein branches joined by a cell-specific fusion gate."""

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        n_latent: int,
        masked_protein_indices: Iterable[int],
        n_cat_list: Iterable[int] | None,
        n_layers: int,
        n_hidden: int,
        dropout_rate: float,
        distribution: str = "normal",
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_proteins = int(n_proteins)
        mask = torch.ones(self.n_proteins)
        mask[list(masked_protein_indices)] = 0.0
        self.register_buffer("protein_input_mask", mask)
        self.rna_encoder = FCLayers(
            n_in=self.n_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )
        self.protein_encoder = FCLayers(
            n_in=self.n_proteins,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )
        self.gate = nn.Sequential(nn.Linear(2 * n_hidden, n_hidden // 2), nn.SiLU(), nn.Linear(n_hidden // 2, 1), nn.Sigmoid())
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = nn.Linear(n_hidden, n_latent)
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)
        self.distribution = distribution
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else _identity
        self.l_transformation = torch.exp
        self.last_gate: torch.Tensor | None = None

    def encode_branches(self, data: torch.Tensor, *cat_list: int):
        rna = data[:, : self.n_genes]
        protein = data[:, self.n_genes : self.n_genes + self.n_proteins]
        protein = protein * self.protein_input_mask
        h_rna = self.rna_encoder(rna, *cat_list)
        h_protein = self.protein_encoder(protein, *cat_list)
        gate = self.gate(torch.cat((h_rna, h_protein), dim=-1))
        return h_rna, h_protein, gate

    def forward(self, data: torch.Tensor, *cat_list: int):
        h_rna, h_protein, gate = self.encode_branches(data, *cat_list)
        self.last_gate = gate.detach()
        fused = gate * h_rna + (1.0 - gate) * h_protein

        qz_m = self.z_mean_encoder(fused)
        qz_v = torch.exp(self.z_var_encoder(fused)) + 1e-4
        q_z = Normal(qz_m, qz_v.sqrt())
        untran_z = q_z.rsample()
        z = self.z_transformation(untran_z)

        ql_m = self.l_gene_mean_encoder(h_rna)
        ql_v = torch.exp(self.l_gene_var_encoder(h_rna)) + 1e-4
        q_l = Normal(ql_m, ql_v.sqrt())
        log_library_gene = torch.clamp(q_l.rsample(), max=15)
        library_gene = self.l_transformation(log_library_gene)
        latent = {"z": z, "l": library_gene}
        untran_latent = {"z": untran_z, "l": log_library_gene}
        return q_z, q_l, latent, untran_latent
