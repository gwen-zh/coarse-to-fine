"""Separate numerical contraction from static-target rehearsal effects.

PDB rehearsal preserves a frozen parent velocity field, rather than forcing
every AA sequence toward its one available crystal conformation. MD examples
still supervise actual individual frames. This is a hypothesis, not a claim
that the parent's ensemble or the resulting model is thermodynamically correct.
"""
import hashlib

import torch

from coarse_few_target_flow import learning_rate
from coarse_full_cohort_flow import training_seed
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case


SOLVERS = [dict(name="euler16", method="euler", steps=16, nfe=16),
           dict(name="midpoint8", method="midpoint", steps=8, nfe=16),
           dict(name="euler64", method="euler", steps=64, nfe=64)]


@torch.no_grad()
def integrate(branch, aa, seed, steps, method="euler"):
    if method not in {"euler", "midpoint"} or steps < 1:
        raise ValueError("Declared method and positive step count required")
    m = (len(aa) + branch.block_size - 1) // branch.block_size
    rng = torch.Generator(device=aa.device).manual_seed(int(seed) % (2 ** 32))
    state = torch.randn((m, branch.coarse_dim), device=aa.device, generator=rng)
    for i in range(steps):
        velocity = branch.coarse_velocity(aa, state, i / steps)
        if method == "midpoint":
            velocity = branch.coarse_velocity(aa, state + velocity / (2 * steps), (i + .5) / steps)
        state = state + velocity / steps
    return branch.denormalize_coarse(state)


def replay_loss(branch, teacher, aa, coarse, generator, domain):
    if domain not in {"PDB", "MD"}:
        raise ValueError("Explicit training domain required")
    target = branch.normalize_coarse(coarse.detach())
    noise = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
    time = torch.rand((), generator=generator, device=target.device, dtype=target.dtype)
    state = (1 - time) * noise + time * target
    # A frozen eval Transformer can otherwise select a different fused CPU
    # kernel from the trainable student. Use the same forward arithmetic in
    # rehearsal; restore the global inference setting even if a call fails.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        velocity = branch.coarse_velocity(aa, state, time)
        if domain == "PDB":
            with torch.no_grad():
                wanted = teacher.coarse_velocity(aa, state, time)
        else:
            wanted = target - noise
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    loss = (velocity - wanted).square().mean()
    digest = hashlib.sha256(noise.detach().cpu().numpy().tobytes() + time.detach().cpu().numpy().tobytes()).hexdigest()
    return loss, dict(draw_sha256=digest, time=float(time), domain=domain)


def update(model, teacher, optimizer, batch, seed, step, p):
    if len(batch) != 4:
        raise ValueError("Exactly3 PDB cases and1 actual MD frame required")
    for case in batch:
        check_case(case, "train")
    model.train(); teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, p["cycle_updates"], p["lr"], p["min_lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    losses = []; draws = []; cases = []
    for slot, case in enumerate(batch):
        domain = "PDB" if slot < 3 else "MD"
        loss, terms = replay_loss(model.coarse_model, teacher, encode_aa(case["sequence"]), case["coarse"],
            torch.Generator().manual_seed(training_seed(seed, step, slot)), domain)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite replay/MD loss")
        (loss / 4).backward(); losses.append(float(loss.detach())); draws.append(terms["draw_sha256"])
        cases.append(dict(protein=case["protein"], case_id=case["id"], domain=domain))
    norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), p["gradient_clip"])
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite replay/MD gradient")
    optimizer.step()
    return dict(update=step + 1, loss=sum(losses) / 4, PDB_replay_loss=sum(losses[:3]) / 3,
        MD_flow_loss=losses[3], gradient_norm=float(norm), batch=cases, base_draws=draws, lr=lr)
