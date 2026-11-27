"""Same frozen SC protocol on the parent hardware, preserving the first attempt.

Only the output directory and allocation placement/concurrency change. All
numeric functions, noise keys, checkpoints, schedules and tolerances are reused
unmodified. The problematic replay is checked before any full recovery run.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_coarse_backbone import sample_coarse_batch
from par_coarse_transfer import assert_frozen,frozen_versions
from par_self_condition_probe import trace_coarse,concatenate_batches
from run_par_coarse_transfer import FeatureBank,verify
import run_par_self_condition_probe as original


FIRST=Path('artifacts/par_self_condition_probe_v1')
RECOVERY=Path('artifacts/par_self_condition_probe_recovery_v1')


def read(path):return json.loads(Path(path).read_text())


def checked_protocol():
    manifest=read(RECOVERY/'recovery_manifest.json')
    if sha(FIRST/'protocol.json')!=manifest['original_protocol_sha256']:
        raise RuntimeError('Original protocol changed')
    for name,digest in manifest['recovery_source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Recovery code changed: '+name)
    p=read(FIRST/'protocol.json');verify(p)
    return p


def precheck(p):
    if platform.node().split('.')[0]!='n041':raise RuntimeError('Parent hardware n041 is required')
    if (RECOVERY/'protocol.json').exists():raise RuntimeError('Recovery already initialized')
    model=original.load_model('axial',2029);guard=frozen_versions(model)
    target=next(t for t in p['targets']if t['protein']=='1ef1_D')
    bank=FeatureBank(p['training_feature_index']);h=bank(target['sequence'])
    batches=[trace_coarse(model,h,90,p['evaluation_noise_keys'][i:i+32],'always',16)for i in [0,32]]
    traced=concatenate_batches(batches)['coarse']
    native=np.concatenate([sample_coarse_batch(model,h,90,p['evaluation_noise_keys'][i:i+32],16).numpy()for i in [0,32]])
    np.savez_compressed(RECOVERY/'replay_precheck.npz',traced=traced,native=native,
                        seeds=np.asarray(p['evaluation_noise_keys'],dtype=np.uint32))
    np.testing.assert_array_equal(traced,native)
    expected=np.load(original.PARENT/'axial/seed2029/step1024/1ef1_D/coarse_nfe16.npz')['coarse']
    np.testing.assert_allclose(native,expected,atol=1e-3,rtol=1e-5)
    assert_frozen(model,guard)
    atomic_json(RECOVERY/'replay_precheck.json',dict(status='passed',host=platform.node(),
        traced_vs_native_exact=True,native_vs_saved_parent_max_abs_A=float(np.abs(native-expected).max()),
        original_atol_A=1e-3,original_rtol=1e-5,optimizer_updates=0,
        weights_unchanged=True,reference_used_for_sampling=False))
    shutil.copyfile(FIRST/'protocol.json',RECOVERY/'protocol.json')
    if sha(RECOVERY/'protocol.json')!=sha(FIRST/'protocol.json'):
        raise RuntimeError('Recovery protocol is not byte-identical')
    print('SC_RECOVERY_PRECHECK_PASSED',float(np.abs(native-expected).max()),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['precheck','run','report'])
    parser.add_argument('--arm',choices=original.ARMS);parser.add_argument('--seed',type=int,choices=original.SEEDS)
    args=parser.parse_args();torch.set_num_threads(2)
    p=checked_protocol()
    if args.stage=='precheck':precheck(p);return
    if read(RECOVERY/'replay_precheck.json')['status']!='passed':raise RuntimeError('Parent-hardware precheck required')
    if sha(RECOVERY/'protocol.json')!=sha(FIRST/'protocol.json'):raise RuntimeError('Recovery protocol changed')
    # Documented I/O-root override only; the frozen numerical functions are
    # invoked directly without editing or replacing their implementations.
    original.ROOT=RECOVERY
    if args.stage=='run':
        if args.arm is None or args.seed is None:parser.error('--arm and --seed required')
        original.run(p,args.seed,args.arm)
    else:original.report(p)


if __name__=='__main__':main()
