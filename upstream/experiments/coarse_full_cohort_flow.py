"""Auditable coarse-only updates for static structures plus native MD frames."""
import hashlib

import numpy as np
import torch

from coarse_few_target_flow import coarse_flow_loss, learning_rate
from fine_geometry_flow import sample_training_batch
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case


def batch_rng(seed):
    return np.random.default_rng(int(seed) + 86417)


def training_seed(seed, step, slot):
    return int(hashlib.sha256(f"full-cohort-coarse-v1:{seed}:{step}:{slot}".encode()).hexdigest()[:8], 16)


def update(model, optimizer, batch, seed, step, p):
    if len(batch) != 4:
        raise ValueError("Three static targets and one native MD frame required")
    for case in batch: check_case(case, "train")
    model.train(); optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, p["cycle_updates"], p["lr"], p["min_lr"])
    for group in optimizer.param_groups: group["lr"] = lr
    losses = []; draws = []; ids = []
    for slot, case in enumerate(batch):
        loss, terms = coarse_flow_loss(model, encode_aa(case["sequence"]), case["coarse"],
            torch.Generator().manual_seed(training_seed(seed, step, slot)))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite full-cohort coarse loss")
        (loss / 4).backward(); losses.append(float(loss.detach())); draws.append(terms["draw_sha256"])
        ids.append(dict(protein=case["protein"], case_id=case["id"], domain="PDB" if slot < 3 else "MD"))
    norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), p["gradient_clip"])
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite full-cohort coarse gradient")
    optimizer.step()
    return dict(update=step + 1, loss=sum(losses) / 4, PDB_loss=sum(losses[:3]) / 3,
        MD_loss=losses[3], gradient_norm=float(norm), batch=ids, base_draws=draws, lr=lr)


def identity_record(row):
    return {key: row[key] for key in ["batch", "base_draws", "lr"]}
