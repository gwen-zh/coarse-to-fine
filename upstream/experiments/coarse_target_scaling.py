"""Nested coarse targets with identical per-target exposure and round LR."""
import hashlib
import math

from coarse_few_target_flow import event


ROUNDS = 256
TARGETS = {"eight": 8, "thirtytwo": 32}


def select_nested_ids(all_ids, first_eight):
    all_ids = list(all_ids); first_eight = list(first_eight)
    if len(set(all_ids)) != len(all_ids) or len(first_eight) != 8 or len(set(first_eight)) != 8 or not set(first_eight) <= set(all_ids):
        raise ValueError("Unique training IDs and eight existing targets required")
    extra = sorted(set(all_ids) - set(first_eight),
        key=lambda name: hashlib.sha256(("coarse-32-training-v1:" + name).encode()).hexdigest())[:24]
    if len(extra) != 24:
        raise ValueError("At least 32 training IDs required")
    return first_eight + extra


def round_learning_rate(step, targets, rounds=ROUNDS, maximum=3e-4, minimum=3e-5):
    if step < 0 or targets < 1 or rounds < 2 or step >= targets * rounds:
        raise ValueError("Step must be inside the declared round budget")
    round_index = step // targets
    return minimum + .5 * (maximum - minimum) * (1 + math.cos(math.pi * round_index / (rounds - 1)))


def plan(arm):
    if arm not in TARGETS:
        raise ValueError("Declare eight or thirtytwo targets")
    count = TARGETS[arm]; total = count * ROUNDS
    return dict(targets=count, rounds=ROUNDS, updates=total,
        milestones=[512, 2048] if count == 8 else [2048, 8192],
        checkpoint_updates=[512, 2048] if count == 8 else list(range(2048, total + 1, 2048)),
        minibatches_per_target=ROUNDS, noise_time_examples_per_target=4 * ROUNDS)


def training_event(step, targets):
    return event(step, targets=targets, batch_size=4)


def initial_metric_roundoff(first, second):
    """Require matching records; permit declared cross-host float roundoff."""
    if isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            raise ValueError("Initial metric keys differ")
        return max((initial_metric_roundoff(first[k], second[k]) for k in first), default=0.)
    if isinstance(first, list):
        if not isinstance(second, list) or len(first) != len(second):
            raise ValueError("Initial metric records differ")
        return max((initial_metric_roundoff(a, b) for a, b in zip(first, second)), default=0.)
    if isinstance(first, float):
        if not isinstance(second, (int, float)) or not math.isfinite(first) or not math.isfinite(second) or not math.isclose(first, second, rel_tol=2e-5, abs_tol=2e-5):
            raise ValueError("Initial metrics differ beyond roundoff")
        return abs(first - second)
    if type(first) is not type(second) or first != second:
        raise ValueError("Initial metric metadata differ")
    return 0.
