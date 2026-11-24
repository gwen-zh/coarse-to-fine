"""Freeze the new stochastic hierarchy and two explicit capacity probes."""
from pathlib import Path
import json
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_pipeline_io import sha
from structural_metrics import aligned_rmsd
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json

RUN=Path('artifacts/hierarchical_ensemble_flow_v1')

def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Use a new protocol version')
    torch.set_num_threads(2)
    parent=Path('artifacts/mixed_hierarchical_generator_v1/protocol.json');tok=json.loads(parent.read_text())['tokenizers']['2026']
    source=Path('artifacts/development/manifest.json');row=next(r for r in json.loads(source.read_text())['all_groups'] if r['id']=='1jfj.A')
    if row['split']!='development' or len(row['paths'])!=2:raise RuntimeError('Expected the already-reused native development pair')
    references=[];paths=[]
    for rel in row['paths']:
        path=Path('upstream')/rel;paths.append(str(path));chain=next(PDBParser(QUIET=True).get_structure(path.stem,str(path)).get_models())['A'];res=[r for r in chain if is_aa(r,standard=True)]
        if ''.join(seq1(r.resname) for r in res)!=row['sequence'] or any(any(a not in r for a in ['N','CA','C']) for r in res):raise RuntimeError('Incomplete same-sequence pair')
        references.append(np.array([[r[a].coord for a in ['N','CA','C']] for r in res],dtype=np.float32))
    decoder=HybridTokenizer.from_checkpoints(tok['warmup'],tok['final']);coarse=[];fine=[];reconstruction=[]
    with torch.no_grad():
        for ref in references:
            c,f,_=decoder.model.encode_latents(torch.from_numpy(ref));ids=decoder.model.fine_book.nearest(f);bb=decoder.decode(c,ids).numpy();coarse.append(c.numpy());fine.append(f.numpy());reconstruction.append(metrics(bb,ref))
    label=RUN/'paired_native_labels.npz';np.savez_compressed(label,coarse=np.stack(coarse),fine_continuous=np.stack(fine),backbone=np.stack(references),sequence=np.array(row['sequence']))
    sources=[str(parent),str(source),str(label),*tok.values(),*paths,*['upstream/experiments/'+x for x in ['hierarchical_ensemble_flow.py','joint_sequence_predictor.py','two_level_autoencoder.py','run_hierarchical_ensemble_probe.py','prepare_hierarchical_ensemble_probe.py','hierarchical_pipeline_io.py','mixed_hierarchical_generator.py','hierarchical_rigid_vq.py','hierarchical_latent_contract.py','structural_metrics.py','compare_two_level_ae_author.py','backbone_quality_audit.py','diagnostic_io.py']], 'upstream/tests/test_hierarchical_ensemble_flow.py','upstream/scripts/hierarchical_ensemble_probe.sbatch']
    p=dict(version=1,seeds=[2026,2027],tokenizer=tok,tokenizer_seed=2026,condition='RawAAIDs->learnedembedding;noESMfeatureinput',sampling_steps=16,coarse_jitter_std=.05,
        architecture='OneTransformerbackbonesharedbycoarsemarginalflowandfineconditionalflow. CoarseM16;fineN8continuous,thenEuclideansnaptoexisting1024book. BlockAAconditioningorderedflattenpool;finebothbroadcastcoarseandattendcoarsetokens. Coarsequeriescannotattendfinestates.',
        objective='Independentlinearlynoisedcoarse/finepaths,velocityMSEtargets. Fineconditionpairednativecoarseplus0.05normalizedjitter;nogeneratedcoarsewithunrelatedfinelabel. Thisislocalrobustnessaugmentation,notguaranteedremovalofgeneratedconditionmismatch.',
        toy=dict(model=dict(width=64,layers=2),updates=2048,lr=.001,min_lr=.0001,samples=128,gate='Both50/50modes>=10percent;meannearestcoarseRMSE<=.25;post-snapfineconsistencywithsampledcoarse>=85percent,bothAAconditions,bothseeds. Analyticdeterministiccoarsemeanis.6frombothmodes;notamatched-computecomparison.'),
        pair=dict(model=dict(width=128,layers=4),updates=2048,lr=.0003,min_lr=.00003,samples=64,native_pair_separation_A=aligned_rmsd(references[0][:,1],references[1][:,1]),native_reconstruction_ca_rmsd_A=[r['ca_rmsd'] for r in reconstruction],native_reconstruction_cn_rates=[r['cn_rate'] for r in reconstruction],gate='Eachnative-modeball(radiuspairRMSD/3)>=10percentsamples;totalnear-native>=50percent;meanCN<=nativeceiling+5pp. Allsamplesretained,noreferenceacceptance.'),
        real_pair_role='1JFJ/1JFK,134residues,samesequence. Onlyexistingmultiple-referencegroupinlocalalready-useddevelopmentmanifest. EXPLICITFIT-ONLYcapacityprobe;thissequencecannotbeclaimedasanindependentevaluationforthismethod. Balanced2conformationweightsaretrainingchoice,notphysicaloccupancies.',
        data_limitation='899singleXraystructurescannotidentifyper-sequencethermodynamicpopulations. LocalBioEmuMDreferencesareprojectedstatistics,notcoordinateframes;BPTItest_var100-modelfileisnotverifiedasnativeMDandisnotused. RealMDtrainingneedsidentifiedcoordinate/topologyfilesandsplitbysequence/trajectory.',
        tests=dict(targeted_unit_tests_passed=12,scope='Wiring/RNG/shape/gradient/frozen-decoderchecks,nottrainedquality'),
        stop_rule='Fixed2048updatesperarm,finalcheckpointonly. Noautomaticbudgetextension,hparamsearch,largertraining,deployment,ordeterministic-baselinepromotion. Fullrealensemblevalidationrequiresactualreferenceensembledata.',
        source_sha256={str(path):sha(path) for path in sources})
    atomic_json(RUN/'protocol.json',p);print('HIERARCHICAL_ENSEMBLE_PROTOCOL_READY',p['pair'],flush=True)

if __name__=='__main__':main()
