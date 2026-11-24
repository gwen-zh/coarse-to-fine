"""No-training shared-frame feasibility check on reused native MD development."""
import json
from pathlib import Path
import numpy as np
import torch
from atlas_shared_frame import align_backbone_to_training_reference, encode_in_supplied_frame
from atlas_decoder_repair import load_repaired_tokenizer
from run_atlas_decoder_repair import load_cache
from compare_two_level_ae_author import metrics
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_shared_frame_probe_v1')
FLOW=Path('artifacts/atlas_ensemble_flow_v1')


@torch.no_grad()
def main():
    torch.set_num_threads(2);RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Use a new shared-frame protocol namespace')
    p=json.loads((FLOW/'protocol.json').read_text());groups=load_cache();reference=groups['R1_train'][0]['target']
    sources=[Path(__file__),Path('upstream/experiments/atlas_shared_frame.py'),Path('upstream/tests/test_atlas_shared_frame.py'),FLOW/'protocol.json',Path('artifacts/atlas_decoder_repair_v1/labels.pt'),Path('artifacts/atlas_decoder_repair_v1/geometry/seed2026/evaluation.json'),Path(p['repair_checkpoint'])]
    atomic_json(RUN/'protocol.json',dict(version=1,training_reference=groups['R1_train'][0]['id'],reference_selection='First R1 training frame, fixed before computing shared-frame reconstruction',groups=['R1_original','R2_original','R2_dense'],weights_changed=False,new_target_convention=True,gates={'CA_increase_vs_repaired_baseline_max_A':.05,'mean_CA_RMSD_max_A':1.,'mean_CN_max':.05,'mean_boundary_CN_max':.15},scope='New candidate label convention checked on reused development frames. No generator training and no original tokenizer replacement. R3 remains sealed.',source_sha256={str(x):sha(x) for x in sources}))
    decoder=load_repaired_tokenizer(**p['tokenizer'],repair_checkpoint=p['repair_checkpoint']);before={k:v.clone() for k,v in decoder.state_dict().items()};rows=[];summary=[];variances={}
    baseline=json.loads(Path('artifacts/atlas_decoder_repair_v1/geometry/seed2026/evaluation.json').read_text())['groups']
    for group in ['R1_original','R2_original','R2_dense']:
        old_c=[];new_c=[];old_f=[];new_f=[];reconstructions=[];records=[]
        for case in groups[group]:
            target=align_backbone_to_training_reference(case['target'],reference);c,f=encode_in_supplied_frame(decoder.model,target);ids=decoder.model.fine_book.nearest(f);bb=decoder.decode(c,ids)
            row=dict(group=group,id=case['id'],metrics=metrics(bb.numpy(),target.numpy()));records.append(row);rows.append(row);reconstructions.append(bb.numpy());old_c.append(case['coarse'].numpy());new_c.append(c.numpy());old_f.append(case['fine_continuous'].numpy());new_f.append(f.numpy())
        average={k:float(np.mean([r['metrics'][k] for r in records])) for k in records[0]['metrics']}
        checks=dict(CA=average['ca_rmsd']<=1.,CA_preserved=average['ca_rmsd']<=baseline[group]['metrics']['ca_rmsd']+.05,CN=average['cn_rate']<=.05,boundary_CN=average['boundary_cn_rate']<=.15)
        summary.append(dict(group=group,cases=len(records),metrics=average,checks=checks,pass_gate=all(checks.values())))
        variance={}
        for level,old,new in [('coarse',old_c,new_c),('fine',old_f,new_f)]:
            a=float(np.stack(old).var(0).mean());b=float(np.stack(new).var(0).mean());variance[level]=dict(original_per_position_variance=a,shared_frame_per_position_variance=b,shared_over_original=b/a)
        variances[group]=variance
        np.savez_compressed(RUN/(group+'.npz'),shared_frame_coarse=np.stack(new_c),shared_frame_fine_continuous=np.stack(new_f),reconstruction=np.stack(reconstructions))
    for key,value in before.items():assert torch.equal(value,decoder.state_dict()[key]),key
    result=dict(status='complete',groups=summary,latent_variance=variances,all_reconstruction_gates_pass=all(r['pass_gate'] for r in summary),weights_unchanged=True,generator_training_started=False,sealed_test_evaluated=False,scope='Feasibility of a new target-coordinate convention on already-used MD development data only; lower latent variance is not proof of better generation.')
    atomic_json(RUN/'report.json',result)
    for r in summary:print('SHARED_FRAME_AUDIT',r['group'],{k:r['metrics'][k] for k in ['ca_rmsd','cn_rate','boundary_cn_rate']},r['checks'],flush=True)
    print('SHARED_FRAME_VARIANCE',json.dumps(variances),flush=True)


if __name__=='__main__':main()
