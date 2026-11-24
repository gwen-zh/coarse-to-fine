"""Pair exact training events across staged and interleaved data curricula."""
import hashlib
import math

import numpy as np

from multisequence_flow_data import check_case


def training_schedule(arm, pretrain_updates, adapt_updates):
    if pretrain_updates < 1 or adapt_updates < 1 or pretrain_updates != 2 * adapt_updates:
        raise ValueError("The matched 3:1 mixture requires pretrain_updates = 2 * adapt_updates")
    if arm not in {"interleaved", "PDB_then_MD"}:
        raise ValueError("Unknown curriculum")
    pdb, md = 0, 0
    rows = []
    for update in range(pretrain_updates + adapt_updates):
        if arm == "interleaved":
            domains = ["PDB", "PDB", "PDB", "MD"]
        elif update < pretrain_updates:
            domains = ["PDB"] * 4
        else:
            domains = ["MD", "MD", "MD", "PDB"]
        events = []
        for domain in domains:
            index = pdb if domain == "PDB" else md
            events.append((domain, index))
            if domain == "PDB":
                pdb += 1
            else:
                md += 1
        rows.append(events)
    return rows


def event_seed(training_seed, domain, index):
    message = f"generator-curriculum-v1:{training_seed}:{domain}:{index}"
    return int(hashlib.sha256(message.encode()).hexdigest()[:8], 16)


def event_pools(groups, training_seed, pretrain_updates, adapt_updates):
    pdb_rng = np.random.default_rng(event_seed(training_seed, "PDB_POOL", 0))
    md_rng = np.random.default_rng(event_seed(training_seed, "MD_POOL", 0))
    pdb_count = 4 * pretrain_updates + adapt_updates
    md_count = 3 * adapt_updates
    pdb = pdb_rng.integers(len(groups["PDB_train"]), size=pdb_count).tolist()
    names = sorted(groups["train"])
    md = []
    for _ in range(md_count):
        name = names[int(md_rng.integers(len(names)))]
        md.append((name, int(md_rng.integers(len(groups["train"][name])))))
    return dict(PDB=pdb, MD=md)


def resolve_event(groups, pools, event):
    domain, index = event
    if domain == "PDB":
        case = groups["PDB_train"][pools[domain][index]]
    elif domain == "MD":
        name, frame = pools[domain][index]
        case = groups["train"][name][frame]
    else:
        raise ValueError("Unknown training event domain")
    check_case(case, "train")
    return case


def learning_rate(p, update):
    """Both curricula use the same LR at every optimizer update."""
    if update < p["pretrain_updates"]:
        step, total = update, p["pretrain_updates"]
        high, low = p["pretrain_lr"], p["pretrain_min_lr"]
    else:
        step, total = update-p["pretrain_updates"], p["adapt_updates"]
        high, low = p["adapt_lr"], p["adapt_min_lr"]
    return low + .5 * (high-low) * (1+math.cos(math.pi*step/max(1, total-1)))
