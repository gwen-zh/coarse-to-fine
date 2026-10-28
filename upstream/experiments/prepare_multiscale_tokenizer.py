"""Freeze bounded development-only two-scale reconstruction protocol."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if (a.run/'protocol.json').exists(): raise RuntimeError('Protocol exists; use a new version instead of overwriting')
    a.run.mkdir(parents=True,exist_ok=True)
    corpus=Path('artifacts/pdb_training_v2')
    rows=json.loads((corpus/'manifest.json').read_text())['targets']
    train=sorted([r for r in rows if r['subset']=='train' and r['length']<=256],key=lambda r:hashlib.sha256(('multiscale20261026'+r['id']).encode()).hexdigest())[:64]
    names=json.loads(Path('artifacts/development/objective_geometry_diagnosis/protocol.json').read_text())['geometry_targets']
    index={r['id']:r for r in rows}; val=[index[n] for n in names]
    assert len(train)==64 and len(val)==24 and all(r['subset']=='validation' for r in val)
    assert not {r['sequence_hash'] for r in train}&{r['sequence_hash'] for r in val}
    targets=[]
    for row in train+val:
        token=corpus/'tokens'/(row['id']+'.npz'); ref=corpus/row['backbone']
        with np.load(token) as z: fine=z['fine'].copy()
        with np.load(ref) as z: bb=z['backbone'].copy()
        assert bb.shape==(row['length'],3,3) and fine.shape==(row['length'],)
        assert np.isfinite(bb).all() and ((fine>=0)&(fine<4096)).all()
        targets.append(dict(target=row['id'],subset=row['subset'],length=row['length'],sequence_hash=row['sequence_hash'],tokens=str(token),reference=str(ref),tokens_sha256=sha(token),reference_sha256=sha(ref)))
    code=['multiscale_tokenizer.py','train_multiscale_tokenizer.py','report_multiscale_tokenizer.py']
    protocol=dict(version=1,seed=20261026,arms=['local','global'],epochs=5,lr=2e-4,train_proteins=64,validation_proteins=24,targets=targets,
      weights='artifacts/tokenizer/decoder_checkpoint_110.pth',codes='artifacts/development/residual_tokens/residual_bridge.pt',
      scope='Bounded single-seed development reconstruction probe using known true structures. Not sequence generation, independent confirmation, ensemble sampling or a three-scale tokenizer. Frozen original fine IDs/codebook and decoder. No58/64 confirmation set access.',
      architecture='Two128wide geometry message-passing blocks; pair SE(3)-invariant distances/local directions/relative frames and clipped sequence offset. Global uses full-chain edges; local uses union of spatial16nearest and sequence+-4 with two-hop propagation. Both start from original quantized128Dfine vectors, then32D EMA-VQ32 bottleneck and identical bounded decoder-embedding residual fusion. This adds no amino-acid prediction model.',
      rate=dict(original_bits_per_residue=12,augmented_bits_per_residue=17,coarse_vocabulary=32,note='Both learned arms have17bits/residue, original baseline12. All token tracks remain lengthN. No compression or same-rate superiority to original fine claimed.'),
      quantizer='Training-only spherical kmeans initialization on first8training proteins;EMAdecay.95,commitment.25,dead-code training restart belowEMAcount.1. Frozen at validation.',
      loss='CA-aligned full backbone mean squared distance plus true C-N bond distance squared error plus VQ commitment. Proper Kabsch rotation detached; frozen decoder differentiated with block checkpointing.',
      fusion='delta_embedding=.1*original_embedding_channel_std*tanh(MLP(normalizedfine,quantizedcoarse)); output layer zero initializes exactly at original embeddings. Decoder only receives embeddings determined by discrete IDs, no coordinates bypass bottleneck.',
      selection='Fixed final epoch5, plus original and epoch0 identity checks; no best-epoch or reference-driven per-sample selection.',
      diagnostics=['original','full','zero_coarse','shuffled_coarse'],
      promotion_gate='Exploratory engineering gate: global final validation meanCA_RMSD at least5percent lower than BOTH local and original; long-rangeCA-distanceMAE no worse than local;CNrate no more than0.5pp above original;at least8/32codesused; shuffledcoarse meanRMSD exceeds intactglobal. Passing only permits considering multi-seed development replication, never immediate independent-test or generator training.',
      fixed_budget='2real-decoder smoke optimization steps per arm; then5epochs x64proteins per arm. No automatic hyperparameter sweep or extension.',
      statistics='Protein-paired bootstrap exploratory95percent intervals,10000replicates; same reused24development proteins, no confirmatory pvalue claims.',
      code_sha256={str(Path('upstream/experiments')/f):sha(Path('upstream/experiments')/f) for f in code},
      source_sha256={str(corpus/'manifest.json'):sha(corpus/'manifest.json')})
    atomic_json(a.run/'protocol.json',protocol); print('MULTISCALE_PROTOCOL_READY',a.run,flush=True)

if __name__=='__main__':main()
