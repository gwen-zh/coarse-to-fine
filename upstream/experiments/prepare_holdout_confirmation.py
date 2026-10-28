"""Lock candidates, endpoints and seeds before holdout model evaluation."""
import argparse
import json
from pathlib import Path
from diagnostic_io import atomic_json
from holdout_confirmation import fingerprint,holdout_targets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    manifest=Path('artifacts/development/manifest.json');targets=holdout_targets(json.loads(manifest.read_text()))
    paths=dict(manifest=manifest,medium_patch=Path('artifacts/development/tokenizer_distillation_ablation/medium_kmeans512.pt'),coarse_adapter=Path('artifacts/development/pretrained_coarse_lora/best_adapter.pt'),semantic_initialization=Path('artifacts/development/pretrained_coarse_lora/semantic_initialization.json'),refiner=Path('artifacts/development/depth_generator/residual/best.pt'),residual_bridge=Path('artifacts/development/residual_tokens/residual_bridge.pt'),source_token_manifest=Path('artifacts/development/residual_tokens/manifest.json'),generator=Path('artifacts/mp_rank_00_model_states.pt'),encoder=Path('artifacts/tokenizer/encoder_checkpoint_110.pth'),decoder=Path('artifacts/tokenizer/decoder_checkpoint_110.pth'))
    for name in ['evaluate_holdout_medium','sample_holdout_coarse','decode_holdout_coarse','holdout_confirmation']:
        paths['code_'+name]=Path('upstream/experiments')/(name+'.py')
    for name in ['coarse_lora','depth_generator','train_pretrained_coarse','tokenizer_distillation_control']:
        paths['source_'+name]=Path('upstream/experiments')/(name+'.py')
    inputs={name:fingerprint(path) for name,path in paths.items()}
    for i,row in enumerate(targets):inputs[f'reference_{i}']=fingerprint(Path('upstream')/row['paths'][0])
    previous=json.loads(Path('artifacts/development/pretrained_coarse_continuation/initial/protocol.json').read_text())
    if inputs['coarse_adapter']['sha256']!=previous['coarse_adapter_sha256'] or inputs['refiner']['sha256']!=previous['refiner_sha256']:raise ValueError('Candidate differs from original development experiment')
    if inputs['source_token_manifest']['sha256']!=previous['token_manifest_sha256']:raise ValueError('Original token semantics changed')
    protocol=dict(version=1,inputs=inputs,targets=targets,chunks=4,device='cpu',dtype='float32',reference='Exactly first path in frozen manifest, one reference per sequence, no closest-reference selection',medium=dict(candidate='Existing kmeans512 and fine-embedding mean lookup, no refitting',baseline='Published medium codebook and lookup; same frozen encoder and decoder',calibration='Original fine reconstruction separately',primary='Mean across58 paired per-protein repaired-minus-original medium CA reconstruction RMSD'),coarse=dict(candidate='Existing selected epoch3 coarse LoRA plus fixed original tiny residual refiner',baseline='Exact untrained semantic-remapped initialization, same refiner and RVQ bridge',samples=2,steps=5,context_seed=731,sample_seed='Existing generation_seed(target_id,sample_index)',context='Original sequence-only sampled_h shared by both arms, generated before semantic remapping/adaptation',primary='Mean across58 paired per-protein adapted-minus-control CA RMSD, first averaging2 samples within protein',hardware='Both arms CPU fp32. Same fixed weights and algorithm as GPU development; hardware/random streams differ, no cross-hardware token equivalence or speed claim'),statistics=dict(bootstrap_replicates=20000,bootstrap_seed=2026,primary_endpoints=2,intervals='95% descriptive and97.5% Bonferroni nominal family-adjusted paired protein bootstrap',secondary='CN rate, CA clash proxy, clashes/residue and2sample spread; no secondary significance claims'),failure_policy='All58 required; nonfinite outputs recorded, no outcome exclusions or successful paired claim with missing primary RMSD. Infrastructure/code fixes logged; no checkpoint,seed,threshold changes after results.',use_policy='One fixed confirmation family, no training or selection on holdout; report either direction',scope='Adaptation-development-disjoint exact-sequence holdout, not homology-isolated or pretraining-unseen; not full ensemble or superiority to original full generator.')
    path=args.run/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Confirmation protocol already locked differently')
    atomic_json(path,protocol);print('HOLDOUT_CONFIRMATION_PROTOCOL_LOCKED',fingerprint(path)['sha256'],flush=True)


if __name__=='__main__':main()
