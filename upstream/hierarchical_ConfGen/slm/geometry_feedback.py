"""Checkpoint-compatible coarse-token repair, with conservative CA diagnostics.

These checks are a sampling heuristic, not a certificate of physical validity.
Inputs are single, continuous chains with coordinates in Angstroms.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class GeometryReport:
    bad: np.ndarray
    missing: int
    broken_bonds: int
    clashes: int
    severity: float

    @property
    def score(self):
        return (self.missing, self.broken_bonds + self.clashes, self.severity)

    def summary(self):
        return dict(missing=self.missing, broken_bonds=self.broken_bonds,
                    clashes=self.clashes, severity=self.severity,
                    bad_residues=int(self.bad.sum()))


@dataclass(frozen=True)
class BackboneReport(GeometryReport):
    peptide_bonds: int
    peptide_severity: float

    def summary(self):
        return dict(super().summary(), peptide_bonds=self.peptide_bonds,
                    peptide_severity=self.peptide_severity)


def assess_ca(ca, bond_min=2.8, bond_max=4.2, clash_min=2.5):
    """Flag gross adjacent-CA distance errors and nonlocal CA overlaps.

    The broad bond interval permits cis peptide geometry. Sequence neighbors
    within two positions are excluded from the overlap check. Missing CA atoms
    are counted explicitly rather than silently passing comparisons with NaN.
    """
    ca = np.asarray(ca, dtype=float)
    if ca.ndim != 2 or ca.shape[1] != 3 or len(ca) == 0:
        raise ValueError("Expected nonempty CA coordinates [N, 3]")
    if not (0 < bond_min < bond_max and 0 < clash_min):
        raise ValueError("Invalid geometry thresholds")
    finite = np.isfinite(ca).all(-1)
    bad = ~finite
    severity = 0.0
    bonds = 0
    clashes = 0
    # Row-wise calculation avoids allocating an N x N x 3 distance tensor.
    for i in range(len(ca)):
        if not finite[i]:
            continue
        if i + 1 < len(ca) and finite[i + 1]:
            d = float(np.linalg.norm(ca[i + 1] - ca[i]))
            if d < bond_min or d > bond_max:
                bad[i:i + 2] = True
                bonds += 1
                severity += max(bond_min - d, d - bond_max)
        js = np.flatnonzero(finite[i + 3:]) + i + 3
        ds = np.linalg.norm(ca[js] - ca[i], axis=-1)
        hit = ds < clash_min
        if hit.any():
            bad[i] = True
            bad[js[hit]] = True
            clashes += int(hit.sum())
            severity += float((clash_min - ds[hit]).sum())
    return GeometryReport(bad, int((~finite).sum()), bonds, clashes, severity)


def repair_mask(bad, radius=2):
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    bad = np.asarray(bad, dtype=bool)
    mask = bad.copy()
    for i in np.flatnonzero(bad):
        mask[max(0, i - radius):min(len(mask), i + radius + 1)] = True
    return mask


def assess_backbone(coordinates):
    """Add peptide C-N continuity to the conservative CA checks.

    This criterion is explicitly optimized by the backbone-feedback variant;
    it must not be described as an independent evaluation metric for that variant.
    """
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 3 or xyz.shape[1] < 3 or xyz.shape[2] != 3:
        raise ValueError("Expected backbone/atom37 coordinates [N,A,3], A>=3")
    report = assess_ca(xyz[:, 1])
    distance = np.linalg.norm(xyz[:-1, 2] - xyz[1:, 0], axis=-1)
    finite = np.isfinite(distance)
    invalid = ~finite | (distance < 1.0) | (distance > 1.8)
    bad = report.bad.copy()
    positions = np.flatnonzero(invalid)
    bad[positions] = True
    bad[positions + 1] = True
    excess = np.maximum(np.maximum(1.0 - distance[finite], distance[finite] - 1.8), 0)
    return BackboneReport(bad, report.missing, report.broken_bonds, report.clashes,
                          report.severity, int(invalid.sum()),
                          float(excess.sum()) + int((~finite).sum()) * 1000.0)


def backbone_improves(candidate, current):
    """Accept Pareto improvements, preventing a CA/C-N count tradeoff."""
    new_counts = (candidate.missing, candidate.broken_bonds + candidate.clashes, candidate.peptide_bonds)
    old_counts = (current.missing, current.broken_bonds + current.clashes, current.peptide_bonds)
    if any(new > old for new, old in zip(new_counts, old_counts)):
        return False
    if any(new < old for new, old in zip(new_counts, old_counts)):
        return True
    new_severity = (candidate.severity, candidate.peptide_severity)
    old_severity = (current.severity, current.peptide_severity)
    return (all(new <= old for new, old in zip(new_severity, old_severity))
            and any(new < old for new, old in zip(new_severity, old_severity)))


@torch.no_grad()
def geometry_feedback(model, sequence_tokens, fine, coarse, lengths, decode_ca,
                      rounds=2, num_steps=5, radius=2, policy="geometry", fixed_budget=False,
                      assessment_fn=None, acceptance_fn=None):
    """Repair one chain at a time, retaining each proposal only if score improves.

    fine/coarse and sequence_tokens include BOS/EOS; decode_ca accepts a single
    fine token sequence without BOS/EOS and returns CA coordinates [N, 3].
    Random control masks the same number of residues as the geometry policy.
    Every proposal re-predicts all fine tokens: decoder attention is nonlocal.
    Trace includes rejected proposals so compute and selection are auditable.
    """
    if rounds < 0 or num_steps < 1 or radius < 0:
        raise ValueError("Invalid repair budget")
    if policy not in ("geometry", "random", "global"):
        raise ValueError("Unknown repair policy")
    if not getattr(model, "noise_removal", True):
        raise ValueError("Feedback requires terminal token prediction (noise_removal=True)")
    if fine.shape != coarse.shape or fine.shape != sequence_tokens.shape:
        raise ValueError("Token tensor shapes must agree")
    fine, coarse = fine.clone(), coarse.clone()
    trace = []
    mask_generator = torch.Generator(device=fine.device)
    mask_generator.manual_seed(torch.initial_seed() ^ 0x5A17)
    for b, length in enumerate(lengths.tolist()):
        if length < 1 or length + 2 > fine.shape[1]:
            raise ValueError("Invalid sequence length")
        seq = sequence_tokens[b:b + 1, :length + 2]
        current_fine = fine[b:b + 1, :length + 2]
        current_coarse = coarse[b:b + 1, :length + 2]

        def assess(tokens):
            ca = decode_ca(tokens[0, 1:length + 1], b)
            if isinstance(ca, torch.Tensor):
                ca = ca.detach().cpu().numpy()
            if assessment_fn is None and np.shape(ca) != (length, 3):
                raise ValueError("Decoded CA length does not match token length")
            if assessment_fn is not None:
                if np.shape(ca)[0] != length:
                    raise ValueError("Decoded backbone length does not match token length")
                return assessment_fn(ca)
            return assess_ca(ca)

        report = assess(current_fine)
        trace.append(dict(sample=b, round=0, policy=policy, **report.summary()))
        for iteration in range(1, rounds + 1):
            mask = repair_mask(report.bad, radius)
            count = int(mask.sum())
            if count == 0 and not fixed_budget:
                break
            if policy == "random":
                mask[:] = False
                indices = torch.randperm(length, device=fine.device,
                                         generator=mask_generator)[:count].cpu().numpy()
                mask[indices] = True
            elif policy == "global":
                mask[:] = True
            prior = current_coarse.clone()
            masked = torch.as_tensor(np.flatnonzero(mask) + 1, device=prior.device)
            prior[:, masked] = model.coarse_mask_index
            started = perf_counter()
            proposal_fine, proposal_coarse = model.ddpm_sample(
                sequence_tokens=seq, num_steps=num_steps, input_prior=prior,
                lengths=lengths[b:b + 1], return_coarse=True)
            # This is essential: unmasked coarse tokens are fixed conditions.
            fixed = torch.as_tensor(np.flatnonzero(~mask) + 1, device=prior.device)
            if not torch.equal(proposal_coarse[:, fixed], current_coarse[:, fixed]):
                raise RuntimeError("Sampler changed fixed coarse tokens")
            proposal_report = assess(proposal_fine)
            accepted = (acceptance_fn(proposal_report, report) if acceptance_fn is not None
                        else proposal_report.score < report.score)
            trace.append(dict(sample=b, round=iteration, policy=policy,
                              masked_residues=int(mask.sum()), accepted=accepted,
                              coarse_steps=num_steps, network_calls=num_steps + 2,
                              elapsed_seconds=perf_counter() - started,
                              **proposal_report.summary()))
            if accepted:
                current_fine, current_coarse = proposal_fine, proposal_coarse
                report = proposal_report
        fine[b, :length + 2] = current_fine[0]
        coarse[b, :length + 2] = current_coarse[0]
    return fine, coarse, trace
