"""Fixed fresh evaluation noise for the training-draw-seed replication."""
import hashlib

from par_coarse_capacity import evaluation_keys


SEEDS = [2027, 2028]


def evaluation_noise_keys():
    keys = [int(hashlib.sha256(f'par-signed-volume-replication-v1:noise:{i}'.encode()).hexdigest()[:8], 16)
            for i in range(64)]
    if len(set(keys)) != 64 or set(keys) & set(evaluation_keys(2026)):
        raise RuntimeError('Replication noise is not fresh and unique')
    return keys
