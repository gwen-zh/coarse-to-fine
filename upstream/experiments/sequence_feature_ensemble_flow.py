"""Capacity-matched AA lookup versus frozen contextual sequence features."""
import torch
from torch import nn

from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from joint_sequence_predictor import AA_ALPHABET, encode_aa
from two_level_autoencoder import pack, positions


class SequenceFeatureEnsembleFlow(HierarchicalEnsembleFlow):
    def __init__(self, *args, feature_mode="AA_lookup", feature_dim=1536, **kwargs):
        if feature_mode not in {"AA_lookup", "ESM3_context"}:
            raise ValueError("Unknown sequence conditioning mode")
        super().__init__(*args, **kwargs)
        width = self.config["width"]
        del self.aa_embedding
        self.sequence_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width))
        rng = torch.Generator().manual_seed(7247)
        self.register_buffer("fixed_aa_lookup", torch.randn(len(AA_ALPHABET), feature_dim, generator=rng))
        self.feature_mode = feature_mode
        self.feature_dim = feature_dim
        self.feature_bank = {}
        self.config.update(feature_mode=feature_mode, feature_dim=feature_dim)

    def attach_feature_bank(self, feature_bank):
        bank = {}
        for sequence, values in feature_bank.items():
            encode_aa(sequence)
            if values.shape != (len(sequence), self.feature_dim) or not torch.isfinite(values).all():
                raise ValueError("Feature sequence mapping or dimensions differ")
            bank[sequence] = values.detach().cpu().clone()
        self.feature_bank = bank

    def aa_conditions(self, aa):
        if aa.ndim != 1 or aa.dtype != torch.long or not len(aa) or aa.min() < 0 or aa.max() >= len(AA_ALPHABET):
            raise ValueError("One valid AA token sequence required")
        if self.feature_mode == "AA_lookup":
            features = self.fixed_aa_lookup[aa]
        else:
            sequence = "".join(AA_ALPHABET[i] for i in aa.cpu().tolist())
            if sequence not in self.feature_bank:
                raise ValueError("Missing AA-only features for this exact sequence; extract them before inference")
            features = self.feature_bank[sequence].to(device=aa.device, dtype=self.coarse_mean.dtype)
        h = self.sequence_projection(features.detach())
        h = h + positions(len(aa), h.device, h.dtype, h.shape[-1])
        packed, valid = pack(h, self.block_size)
        block = self.block_condition(torch.cat([packed.flatten(1), valid.to(h.dtype)], -1))
        return h, block
