"""Reference-free independent proposals within a fixed token-edit budget."""
import numpy as np


def maximum_edits(length, fraction):
    if length < 1 or not 0 <= fraction <= 1:
        raise ValueError('Invalid sequence length or edit cap')
    count = int(np.floor(length * fraction))
    while count > 0 and count / length > fraction:
        count -= 1
    while count < length and (count + 1) / length <= fraction:
        count += 1
    return count


def bounded_proposals(topk, original, count, seed, edit_cap, distribution):
    """Every proposal is based on original, never on accepted candidates.

    Uniform edit counts cover the full cap. Geometric counts give probability
    proportional to 2**(-m), m=1..cap. Residues are uniform without replacement;
    alternatives are uniform over fixed top-k excluding the initial token.
    A zero feasible cap yields unchanged proposals, counted as padding.
    """
    table = np.asarray(topk)
    initial = np.asarray(original)
    if table.ndim != 2 or initial.shape != (len(table),) or count < 0:
        raise ValueError('Invalid proposal dimensions/count')
    if distribution not in ('uniform', 'geometric'):
        raise ValueError('Unknown edit-count distribution')
    alternatives = []
    for row, token in zip(table, initial):
        if len(set(row.tolist())) != len(row) or token not in row:
            raise ValueError('Each fixed pool must be unique and contain origin')
        choices = row[row != token]
        if len(choices) == 0:
            raise ValueError('No alternative tokens')
        alternatives.append(choices)
    cap = maximum_edits(len(initial), edit_cap)
    if cap == 0:
        return [(np.array([], dtype=int), np.array([], dtype=initial.dtype)) for _ in range(count)]
    sizes = np.arange(1, cap + 1)
    weights = np.exp2(-sizes.astype(float)) if distribution == 'geometric' else np.ones(cap)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    stream = []
    for _ in range(count):
        size = int(rng.choice(sizes, p=weights))
        sites = rng.choice(len(initial), size=size, replace=False)
        tokens = np.asarray([rng.choice(alternatives[i]) for i in sites], dtype=initial.dtype)
        stream.append((sites, tokens))
    return stream


def apply_multiedit(original, proposal):
    result = np.asarray(original).copy()
    sites, tokens = proposal
    result[sites] = tokens
    return result
