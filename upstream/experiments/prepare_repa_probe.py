import argparse,hashlib,json
from pathlib import Path
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(exist_ok=True,parents=True)
    names=json.loads(Path('artifacts/development/objective_geometry_diagnosis/protocol.json').read_text())['geometry_targets'];sources=['artifacts/pdb_training_v2/native_tokens/manifest.json','artifacts/pdb_training_v2/tokens/manifest.json','artifacts/development/latent_sensitivity_v1/calibration.npz'];code=['repa_alignment.py','train_repa_probe.py']
    protocol=dict(version=1,scope='Exploratory original-token fine-stage REPA-style alignment versus matched CE-only adapters, one seed;899train/101 reused validation;24 reused validation proteins x8 draws. No58/64 independent test access. Original sequence-only coarse sampled before adapters; fine-only invocation prevents coarse-policy drift. Single-structure supervision can reduce diversity.',checkpoint='artifacts/mp_rank_00_model_states.pt',tokens='artifacts/pdb_training_v2/native_tokens',latent_tokens='artifacts/pdb_training_v2/tokens',prefixes='artifacts/development/native_generator/initial_prefixes',calibration=sources[-1],geometry_targets=names,epochs=5,lr=1e-5,alignment_weight=.1,rank=8,seed=20261026,selection='Lowest fine CE on101 reused validation proteins conditioned on original cached generated coarse, with epoch0 rollback eligible. Same selection and train order both arms; no reference geometry selection.',conditioning='50percent true coarse /50percent original cached sampled coarse. Both CE and alignment conditioned on same draw. Static true latent supervision remains potentially ambiguous under wrong sampled coarse.',alignment='One predeclared block near one-third depth, centered/scaled training-only raw encoder latent cosine target. Same attention LoRA locations and final fine-output LoRA in both arms; projector discarded for inference. Not an exact replication of image REPA.',code_sha256={str(Path('upstream/experiments')/f):hashlib.sha256((Path('upstream/experiments')/f).read_bytes()).hexdigest() for f in code},source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources})
    atomic_json(a.run/'protocol.json',protocol);print('REPA_PROTOCOL_READY',flush=True)


if __name__=='__main__':main()
