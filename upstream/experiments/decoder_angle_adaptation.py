"""Decoder-only continuous-latent adaptation with a matched angular objective."""
import hashlib
import math

import torch
from torch.nn import functional as F

from atlas_decoder_repair import repair_loss
from multisequence_flow_data import check_case


ARMS = ["continuation", "angular"]
DECODER_PREFIX = "fine_tokenizer.decoder."


def enable_decoder(codec):
    codec.eval().requires_grad_(False)
    codec.fine_tokenizer.decoder.train().requires_grad_(True)
    return list(codec.fine_tokenizer.decoder.parameters())


def frozen_digest(codec):
    digest = hashlib.sha256()
    for name, value in codec.state_dict().items():
        if not name.startswith(DECODER_PREFIX):
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def angles(x):
    def angle(a, b):
        cosine = (F.normalize(a, dim=-1, eps=1e-6) * F.normalize(b, dim=-1, eps=1e-6)).sum(-1)
        return torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
    return torch.stack([
        angle(x[:-1, 1] - x[:-1, 2], x[1:, 0] - x[:-1, 2]),
        angle(x[:-1, 2] - x[1:, 0], x[1:, 1] - x[1:, 0])], -1)


def angular_term(prediction, target, scale_degrees=5.):
    """Squared radians normalized by5deg; balance all and block-boundary bonds."""
    if scale_degrees <= 0 or prediction.shape != target.shape or len(target) < 5:
        raise ValueError("Matching chains and positive angle scale required")
    error = ((angles(prediction) - angles(target.detach())) / math.radians(scale_degrees)).square()
    ids = torch.arange(len(target) - 1, device=target.device)
    boundary = ids // 4 != (ids + 1) // 4
    return .5 * (error.mean() + error[boundary].mean())


def objective(prediction, target, arm, p):
    if arm not in ARMS:
        raise ValueError("Undeclared decoder arm")
    base, terms = repair_loss(prediction, target, geometry=True)
    angular = angular_term(prediction, target, p["angle_scale_degrees"])
    weight = p["angle_weight"] if arm == "angular" else 0.
    return base + weight * angular, dict(base=base.detach(), angular=angular.detach(), **terms)


def update(codec, optimizer, batch, arm, step, p):
    if len(batch) != 4:
        raise ValueError("Three PDB examples plus one native MD frame required")
    codec.eval(); codec.fine_tokenizer.decoder.train(); optimizer.zero_grad(set_to_none=True)
    lr = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / max(1, p["updates"] - 1)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    recorded = {}; total = 0.; ids = []
    for slot, case in enumerate(batch):
        check_case(case, "train")
        target = case["target"] - case["target"][:, 1].mean(0)
        with torch.no_grad():
            context = codec.context(case["coarse"], len(target))
        # The inference convenience codec.decode() is no_grad: bypass it here.
        predicted = codec.fine_tokenizer.decode_vectors(case["fine_continuous"].detach(), context, len(target))
        loss, terms = objective(predicted, target, arm, p)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite decoder adaptation loss")
        (loss / 4).backward(); total += float(loss.detach()) / 4
        for key, value in terms.items():
            recorded[key] = recorded.get(key, 0.) + float(value) / 4
        ids.append(dict(case_id=case["id"], protein=case["protein"], domain="PDB" if slot < 3 else "MD"))
    norm = torch.nn.utils.clip_grad_norm_(codec.fine_tokenizer.decoder.parameters(), p["gradient_clip"])
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite decoder gradient")
    optimizer.step()
    return dict(update=step + 1, batch=ids, lr=lr, loss=total, gradient_norm=float(norm), **recorded)
