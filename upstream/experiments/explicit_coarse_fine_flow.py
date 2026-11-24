"""Zero-initialized, capacity-matched coarse conditioning for fine flow.

The local_geometry arm exposes exactly the deterministic 24-dimensional
coarse-derived geometric context already used by the frozen fine decoder.
raw_context uses normalized XYZ padded to 24 dimensions in the same adapter.
Neither arm accesses native fine labels or a reference structure at inference.
"""
import torch
from torch import nn
from torch.nn import functional as F

from block_stable_frame_tokenizer import stable_frame_context
from hierarchical_ensemble_flow import fine_attention_mask
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow


class ExplicitCoarseFineFlow(SequenceFeatureEnsembleFlow):
    def __init__(self, *args, context_kind, **kwargs):
        super().__init__(*args, **kwargs)
        if self.coarse_dim != 3 or context_kind not in {"raw_context", "local_geometry"}:
            raise ValueError("Physical centers and a declared context arm required")
        self.context_kind = context_kind
        width = self.config["width"]
        self.context_adapter = nn.Sequential(nn.Linear(24, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.context_adapter[-1].weight)
        nn.init.zeros_(self.context_adapter[-1].bias)
        self._geometry_cache = None

    def context_features(self, condition, length):
        self._coarse_state(condition, length)
        if self.context_kind == "raw_context":
            return F.pad(condition, (0, 21))
        cached = self._geometry_cache
        if (not condition.requires_grad and cached is not None
                and cached[0].device == condition.device and cached[0].dtype == condition.dtype
                and torch.equal(cached[0], condition)):
            return cached[1]
        centers = self.denormalize_coarse(condition)
        features = stable_frame_context(centers, length, "local", self.block_size)["geometry"]
        if not condition.requires_grad:
            self._geometry_cache = (condition.detach().clone(), features.detach())
        return features

    def fine_hidden(self, aa, state, t, coarse_condition):
        residue, block = self.aa_conditions(aa)
        self._coarse_state(coarse_condition, len(aa))
        if state.shape != (len(aa), self.fine_dim) or not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("Finite matching fine-state shape required")
        extra = self.context_adapter(self.context_features(coarse_condition, len(aa)))
        time = self.time_state(t, residue)
        mode = self.mode.weight[1]
        c = block + self.coarse_input(coarse_condition) + self.kind.weight[0] + mode + time
        c = c + extra
        local = coarse_condition.repeat_interleave(self.block_size, dim=0)[:len(aa)]
        f = residue + self.fine_input(state) + self.coarse_to_fine(local) + self.kind.weight[1] + mode + time
        f = f + extra.repeat_interleave(self.block_size, dim=0)[:len(aa)]
        mask = fine_attention_mask(len(c), len(f), state.device)
        return self.transformer(torch.cat([c, f])[None], mask=mask)[0]
