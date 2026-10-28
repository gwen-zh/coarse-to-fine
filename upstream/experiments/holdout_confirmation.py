"""Locked holdout metadata and protein-level confirmation statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np


def fingerprint(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    stat=path.stat()
    return dict(path=str(path),sha256=digest.hexdigest(),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def holdout_targets(manifest):
    targets=sorted([r for r in manifest['all_groups'] if r['split']=='holdout'],key=lambda r:r['sequence_hash'])
    development={r['sequence_hash'] for r in manifest['all_groups'] if r['split']=='development'}
    if len(targets)!=58 or len({r['sequence_hash'] for r in targets})!=58:raise ValueError('Expected58 unique holdout sequences')
    for row in targets:
        if hashlib.sha256(row['sequence'].encode()).hexdigest()!=row['sequence_hash']:raise ValueError('Sequence hash mismatch')
        if len(row['sequence'])!=row['length'] or not row['paths'] or row['sequence_hash'] in development:raise ValueError('Invalid holdout membership')
    return targets


def load_locked(run):
    path=Path(run)/'protocol.json';protocol=json.loads(path.read_text())
    for role,entry in protocol['inputs'].items():
        stat=Path(entry['path']).stat()
        if stat.st_size!=entry['size'] or stat.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Locked input changed: '+role)
    return protocol,hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(differences,seed=2026):
    values=np.asarray(differences,float)
    if values.shape!=(58,) or not np.isfinite(values).all():raise ValueError('All58 finite paired protein outcomes required')
    rng=np.random.default_rng(seed);means=values[rng.integers(0,58,size=(20000,58))].mean(1)
    interval=np.quantile(means,[.0125,.9875]).tolist()
    return dict(proteins=58,mean_difference=float(values.mean()),median_difference=float(np.median(values)),proteins_improved=int((values<0).sum()),bootstrap95=np.quantile(means,[.025,.975]).tolist(),bootstrap97_5=interval,negative_family_adjusted_interval=bool(interval[1]<0),note='Paired protein bootstrap;97.5% interval uses Bonferroni nominal coverage for two prespecified primary RMSD endpoints; no exact finite-sample coverage guarantee.')
