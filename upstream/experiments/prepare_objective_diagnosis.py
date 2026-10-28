"""Lock A/C protocols before running new diagnostic measurements; no holdout access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fine=Path('artifacts/pdb_training_v2/fine_retention_features')
    tail=Path('artifacts/pdb_training_v2/residual_condition_features')
    corpus=Path('artifacts/pdb_training_v2')
    mid=Path('artifacts/development/residual_condition/residual')
    manifests={name:json.loads((path/'manifest.json').read_text()) for name,path in [('fine',fine),('tail',tail)]}
    rows={name:{r['id']:r for r in m['targets'] if r['dataset']=='corpus'} for name,m in manifests.items()}
    order=lambda name:hashlib.sha256(name.encode()).hexdigest()
    train=sorted([name for name,r in rows['tail'].items() if r['subset']=='train'],key=order)
    validation=sorted([name for name,r in rows['tail'].items() if r['subset']=='validation'],key=order)
    geometry=validation[:24]
    if len(train)!=899 or len(validation)!=101 or len(geometry)!=24:raise ValueError('Unexpected corpus split')
    old=json.loads(Path('artifacts/development/geometry_condition/no_geometry/condition_protocol.json').read_text())
    if old['geometry_targets']!=geometry:raise ValueError('Geometry target selection changed')
    for name in geometry:
        if rows['fine'][name]['subset']!='validation':raise ValueError('Split mismatch')
        with np.load(fine/rows['fine'][name]['features']) as a,np.load(tail/rows['tail'][name]['features']) as b:
            for key in ['coarse','fine']:
                if not np.array_equal(a[key],b[key]):raise ValueError('A/C original context differs')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=dict(version=1,train_targets=train,validation_targets=validation,geometry_targets=geometry,
        fine_cache=str(fine),tail_cache=str(tail),corpus=str(corpus),mid_source=str(mid),
        hashes=dict(fine_manifest=sha(fine/'manifest.json'),tail_manifest=sha(tail/'manifest.json'),
                    corpus_manifest=sha(corpus/'manifest.json'),mid_checkpoint=sha(mid/'mid.pt'),
                    fine_projection=sha(fine/'fine_projection.pt')),
        A=dict(temperatures=[.5,1.,2.],samples_per_temperature=16,seed=2026,chunks=4,
            score='Mean log p_T=1(fine token | same original cached sampled coarse,sequence), conditional factorized fine-head score; NOT full diffusion likelihood',
            controls=['original argmax','reference-encoded original fine tokens decoded by same decoder','experimental reference backbone geometry'],
            primary='Mean per-protein Spearman(score,CN outlier rate) within T=1; protein bootstrap, undefined correlations explicit',
            secondary='Within each temperature separately: score vs CA clashes/residue; groundtruth reconstruction and argmax; no pooled-temperature primary correlation'),
        C=dict(seeds=[0,1,2],arms=['ce','ce_kl'],epochs=5,lr=1e-4,progress=[0.,.1,.5,1.,3.,5.],
            snapshot_steps=[0,90,450,899,2697,4495],beta=dict(ce=0.,ce_kl=1.),
            train_proteins=899,validation_proteins=101,train_ce_targets=train[:16],
            initialization='Same zero condition projection and frozen original tail; seeds vary training order only',
            control='Epoch0 original output repeated as fixed no-update control at every progress, verified against original cached fine',
            primary='Final-minus-initial CN outlier-rate per protein, mean across3 seeds; CE vs CE+KL paired difference; protein bootstrap conditional on these seeds',
            secondary='Whole predetermined trajectory, no checkpoint selection; report seed curves and monotonicity descriptively, never require monotonicity'),
        scope='Repeated development diagnosis on existing new corpus, not independent confirmation or proof of universal causal mechanism. Holdout58 untouched. No model selection or geometry hyperparameter sweep.')
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Locked protocol differs')
    atomic_json(path,protocol)
    print('OBJECTIVE_DIAGNOSIS_PROTOCOL_LOCKED',sha(path),flush=True)


if __name__=='__main__':main()
