"""Model arms for the missing-protein benchmark and their construction.

Every arm builds the stock scvi-tools TOTALVI with the configured decoder
width (`totalvi_hidden`) and then swaps only the encoder, so the decoder,
likelihoods and background model are identical across arms.

Arms are declared in config/paper_benchmark.yaml under `arms:`:
    encoder: joint | joint_available | fusion
    hidden:  encoder width (int)
    gate:    learned | fixed | rna_only   (fusion only)
"""

from __future__ import annotations

from scvi.model import TOTALVI
from scvi.nn import EncoderTOTALVI

from fusionvi import AvailabilityAwareJointEncoderTOTALVI, FusionVIEncoder, ModalityDropoutJointEncoderTOTALVI


def build_model(adata, cfg: dict, arm: dict) -> TOTALVI:
    model = TOTALVI(
        adata,
        n_latent=int(cfg["n_latent"]),
        n_hidden=int(cfg["totalvi_hidden"]),
        n_layers_encoder=2,
        n_layers_decoder=1,
        gene_likelihood="nb",
        latent_distribution="normal",
        empirical_protein_background_prior=False,
    )
    n_genes = adata.n_vars
    n_proteins = adata.obsm["protein_counts"].shape[1]
    batch_categories = list(adata.obs["batch"].cat.categories)
    available_batches = cfg.get("panel_available_batches") or [cfg["source_batch"]]
    available_batch_indices = [
        batch_categories.index(batch)
        for batch in available_batches
        if batch in batch_categories
    ]
    common = dict(
        n_latent=int(cfg["n_latent"]),
        n_cat_list=[model.module.n_batch],
        n_layers=2,
        n_hidden=int(arm["hidden"]),
        dropout_rate=0.2,
        distribution="normal",
    )
    kind = arm["encoder"]
    if kind == "joint":
        if int(arm["hidden"]) != int(cfg["totalvi_hidden"]):
            enc = {k: v for k, v in common.items() if k != "n_latent"}
            model.module.encoder = EncoderTOTALVI(
                n_genes + n_proteins, int(cfg["n_latent"]), use_batch_norm=True, **enc
            )
    elif kind == "joint_available":
        model.module.encoder = AvailabilityAwareJointEncoderTOTALVI(
            n_genes=n_genes,
            n_proteins=n_proteins,
            available_batch_indices=available_batch_indices,
            **common,
        )
    elif kind == "joint_moddrop":
        model.module.encoder = ModalityDropoutJointEncoderTOTALVI(
            n_genes=n_genes,
            n_proteins=n_proteins,
            modality_dropout=float(arm.get("modality_dropout", 0.3)),
            available_batch_indices=available_batch_indices,
            **common,
        )
    elif kind == "fusion":
        model.module.encoder = FusionVIEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            masked_protein_indices=[],
            gate_mode=arm.get("gate", "learned"),
            modality_dropout=float(arm.get("modality_dropout", 0.0)),
            available_batch_indices=available_batch_indices,
            **common,
        )
    else:
        raise ValueError(f"Unknown encoder kind {kind!r}")
    return model


def n_trainable(model: TOTALVI) -> int:
    return int(sum(p.numel() for p in model.module.parameters() if p.requires_grad))
