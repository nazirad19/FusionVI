"""Encoders for the paper totalVI baseline and the final FusionVI model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from scvi.nn import EncoderTOTALVI, FCLayers


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _panel_available(
    protein: torch.Tensor,
    cat_list: tuple,
    available_batch_indices: tuple[int, ...] | None,
) -> torch.Tensor:
    """Return an explicit per-cell panel-availability mask.

    The paper benchmark supplies batch as the first categorical covariate.
    Using that metadata avoids confusing a genuinely all-zero measured cell
    with an experimentally absent panel. The count-based fallback is retained
    only for callers that do not provide availability metadata.
    """
    if available_batch_indices is not None and cat_list:
        batch = cat_list[0].reshape(-1, 1).long()
        available = torch.zeros_like(batch, dtype=torch.bool)
        for index in available_batch_indices:
            available |= batch == int(index)
        return available.to(protein.dtype)
    return (protein.sum(dim=-1, keepdim=True) > 0).to(protein.dtype)


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


class AvailabilityAwareJointEncoderTOTALVI(nn.Module):
    """Joint-encoder control that is explicitly told when proteins are absent.

    Cells without a measured protein panel receive a learned placeholder and
    a binary availability indicator. This separates missing-input handling
    from the effect of FusionVI's two branches and gate.
    """

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        n_latent: int,
        n_cat_list: Iterable[int] | None,
        n_layers: int,
        n_hidden: int,
        dropout_rate: float,
        distribution: str = "normal",
        available_batch_indices: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_proteins = int(n_proteins)
        self.available_batch_indices = None if available_batch_indices is None else tuple(int(x) for x in available_batch_indices)
        self.missing_protein_embedding = nn.Parameter(torch.zeros(self.n_proteins))
        self.base = EncoderTOTALVI(
            self.n_genes + self.n_proteins + 1,
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
        available = _panel_available(protein, cat_list, self.available_batch_indices).to(data.dtype)
        protein = available * protein + (1.0 - available) * self.missing_protein_embedding
        return self.base(torch.cat((rna, protein, available, remainder), dim=-1), *cat_list)


class ModalityDropoutJointEncoderTOTALVI(nn.Module):
    """Stock joint encoder trained with supervised whole-panel dropout.

    A random subset of source cells has its protein encoder input zeroed while
    the original tensor still reaches totalVI's protein likelihood. Thus the
    decoder receives protein-supervised gradients from RNA-only latents.
    """

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        n_latent: int,
        n_cat_list: Iterable[int] | None,
        n_layers: int,
        n_hidden: int,
        dropout_rate: float,
        distribution: str = "normal",
        modality_dropout: float = 0.3,
        available_batch_indices: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.n_genes = int(n_genes)
        self.n_proteins = int(n_proteins)
        self.modality_dropout = float(modality_dropout)
        self.available_batch_indices = None if available_batch_indices is None else tuple(int(x) for x in available_batch_indices)
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
        self.last_panel_dropped: torch.Tensor | None = None

    def forward(self, data: torch.Tensor, *cat_list: int):
        rna = data[:, : self.n_genes]
        protein = data[:, self.n_genes : self.n_genes + self.n_proteins]
        remainder = data[:, self.n_genes + self.n_proteins :]
        available = _panel_available(protein, cat_list, self.available_batch_indices).bool()
        dropped = torch.zeros_like(available)
        if self.training and self.modality_dropout > 0:
            dropped = available & (torch.rand_like(available, dtype=torch.float32) < self.modality_dropout)
            protein = protein * (~dropped).to(protein.dtype)
        self.last_panel_dropped = dropped.detach()
        return self.base(torch.cat((rna, protein, remainder), dim=-1), *cat_list)


GATE_MODES = ("learned", "fixed", "rna_only")


class FusionVIEncoder(nn.Module):
    """Separate branches with a gate that respects protein-panel availability.

    The learned gate is used when protein measurements are present.  When a
    cell has no measured proteins, fusion is forced to the RNA branch.  This
    prevents an all-zero placeholder panel, together with encoder biases and a
    batch covariate, from contributing a spurious protein representation.
    """

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
        gate_mode: str = "learned",
        fixed_gate: float = 0.5,
        modality_dropout: float = 0.0,
        available_batch_indices: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        self.gate_mode = gate_mode
        self.fixed_gate = float(fixed_gate)
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.modality_dropout = float(modality_dropout)
        self.available_batch_indices = None if available_batch_indices is None else tuple(int(x) for x in available_batch_indices)
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
        panel_available = _panel_available(protein, cat_list, self.available_batch_indices)
        dropped = torch.zeros_like(panel_available, dtype=torch.bool)
        if self.training and self.modality_dropout > 0:
            dropped = panel_available.bool() & (torch.rand_like(panel_available) < self.modality_dropout)
            protein = protein * (~dropped).to(protein.dtype)
        h_rna = self.rna_encoder(rna, *cat_list)
        h_protein = self.protein_encoder(protein, *cat_list)
        learned_gate = self.gate(torch.cat((h_rna, h_protein), dim=-1))
        if self.gate_mode == "fixed":
            learned_gate = torch.full_like(learned_gate, self.fixed_gate)
        elif self.gate_mode == "rna_only":
            learned_gate = torch.ones_like(learned_gate)
        protein_available = panel_available * (~dropped).to(panel_available.dtype)
        gate = 1.0 - protein_available * (1.0 - learned_gate)
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
