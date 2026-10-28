import argparse,json,hashlib
from pathlib import Path
from diagnostic_io import atomic_json
from prepare_half_fine_tokenizer import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if (a.run/'protocol.json').exists():raise RuntimeError('Protocol already exists')
    a.run.mkdir(parents=True,exist_ok=True)
    parent=Path('artifacts/development/half_fine_tokenizer_v1');old=json.loads((parent/'protocol.json').read_text())
    rows=[r for r in old['targets'] if r['subset']=='validation'];assert len(rows)==24
    inputs={str(parent/'protocol.json'):sha(parent/'protocol.json')}
    for arm in ('local','global'):
        paths=[parent/arm/'final.pt',parent/arm/'evaluation.json']+[parent/arm/(r['target']+'.npz') for r in rows]
        inputs.update({str(p):sha(p) for p in paths})
    source=['half_fine_decode_diagnostic.py','run_half_fine_decode_diagnostic.py','report_half_fine_decode_diagnostic.py','half_fine_tokenizer.py','multiscale_tokenizer.py','evaluate_residual_vq.py','train_multiscale_tokenizer.py','structural_metrics.py','backbone_quality_audit.py','report_multiscale_tokenizer.py']
    protocol=dict(version=1,parent=str(parent),parent_protocol_sha256=sha(parent/'protocol.json'),targets=rows,
       weights=old['weights'],codes=old['codes'],arms=['local','global'],variants=['continuous','visible_fixed','hidden_snapped'],
       scope='No-training decode intervention on saved final half-fine checkpoints and the same24reuseddevelopment proteins. True-structure reconstruction, not sequence generation or independent validation. No58/64confirmation data.',
       fixed_inputs='Use the exact saved per-protein coarseIDs from parent evaluation and checkpoint EMAcodevectors. Fusion receives only retainedfineIDs and thesecoarseIDs; no re-encoding, changed coarse choices or hiddenfine labels as inputs. Original fullfine is decoded as a separate reconstruction reference only.',
       intervention='A current continuous fusion;B restore visible positions exactly to originalfine embeddings with hidden predictions unchanged;C sameB plus hidden-only nearest original4096decoder embeddings. Only ordinary fine vocabulary; no specialtokens. All three variants run through same frozen decoder on sameGPU.',
       distance='Squared Euclidean after per-channel scaling by std of original4096decoderembedding table, matching parent hiddenembedding supervision.1280D decoder space, not128D encoder codebook. Fixed nearest1, no metric/k/temperature/reference-based selection.',
       accounting='168 decodercalls=24*(original+2arms*3variants). No additional training or proposal search. Local/global have same transmitted11-ishbits budget; snapped output IDs are deterministic expansion, not additionally transmitted bits.',
       selection='No per-sample fallback or choosing an intervention by reference. Report all variants, all proteins. Hiddenfine accuracy and true coordinates only used in posthoc metrics.',
       next_step_gate='Consider a new discrete completion training pilot only if global snapped CNrate <=20percent and <=half globalcontinuousCNrate, globalsnapped referenceRMSD <=1.10*globalcontinuousRMSD, globalsnapped meanRMSD <=.95*localsnapped and paired95percentupperbound<0. This is an engineering screen, not confirmed efficacy; passing does not authorize independent-test model selection.',
       replay='Record coordinate and metric differences of rerun continuous versus cached parent, and confirm savedcoarse IDs/retained mask. Do not silently assume crossGPU bitwise coordinate equality. Stop interpretation if maxpairedcontinuousCA_RMSDdifference>0.05A or maxCNrate difference>0.01; investigate rather than relax.',
       statistics='Pairedprotein descriptive bootstrap10000replicates, fixedseed20261026. Reuseddevelopment/singleseed; no confirmatory claim.',
       code_sha256={str(Path('upstream/experiments')/f):sha(Path('upstream/experiments')/f) for f in source},
       source_sha256={**old['source_sha256'],**inputs})
    atomic_json(a.run/'protocol.json',protocol);print('HALF_FINE_DECODE_PROTOCOL_READY',flush=True)

if __name__=='__main__':main()
