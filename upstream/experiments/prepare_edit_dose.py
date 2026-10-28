"""Freeze exact candidate edit counts without changing the acceptance constraints."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from bounded_multiedit_proposals import maximum_edits
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'protocol.json').exists():raise FileExistsError('Never overwrite locked dose protocol')
    parent=Path('artifacts/development/bounded_multiedit_v1/protocol.json');old=json.loads(parent.read_text());targets=old['targets'];doses=[1,2,4,8];lengths=[]
    assert len(targets)==192 and len({x['protein'] for x in targets})==24 and sum(x['decoder_call_budget'] for x in targets)==8625
    for item in targets:
        path=Path(old['source'])/item['input'];assert hashlib.sha256(path.read_bytes()).hexdigest()==old['input_sha256'][item['target']]
        with np.load(path) as z:
            n=len(z['fine']);lengths.append(n);assert maximum_edits(n,.1)>=max(doses)
            assert np.array_equal(z['logits'].argmax(-1),z['fine'])
    files=['edit_dose_proposals.py','run_edit_dose.py','report_edit_dose.py','prepare_edit_dose.py','bounded_multiedit_proposals.py','fine_search_constraints.py','structural_metrics.py','backbone_quality_audit.py','evaluate_residual_vq.py','diagnostic_io.py']
    paths=[Path('upstream/experiments')/f for f in files]+[Path('upstream/scripts')/('edit_dose_'+suffix+'.sbatch') for suffix in ['preflight','search','report']]
    protocol=dict(version=1,scope='Exploratory candidate edit-count sensitivity on24reused development proteins x8original outputs; no new method selection or independent confirmation. No58/64cohort access, no training.',source=old['source'],corpus=old['corpus'],weights=old['weights'],targets=targets,input_sha256=old['input_sha256'],parent_protocol_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),doses=doses,arms=['edits_'+str(k) for k in doses],chunks=16,delta_angstrom=1.,edit_fraction_cap=.1,proposal_seed=20261027,bootstrap_seed=20261027,bootstrap_draws=20000,
      proposal='Each candidate starts from original; exact1/2/4/8unique sites changed to uniform original-headtop5 alternatives excluding original token. Everycall samples one ordered8site subset uniformlywithoutreplacement and8alternatives; smaller doses use prefixes, giving pairednestedproposals and correct uniform marginals. Independent of acceptance and references.',
      budget='Same per-output calls as previous lockedexperiment;8625decoder calls per arm. Retain zero-budgetoutputs androllbacks, duplicateproposals counted. CPU16threads, all4arms samehost perinput, rotatingarmorder. Not equal totalFLOPs or a sweep of call budget.',
      selection='Unchanged initial-relative CN/CA gate with at leastonestrictimprovement,<=1AalignedCAdisplacement,<=10percentedits;lexicographic CNthenCA; rollback if no candidate qualifies. References loaded after everydose completes. Final editcount is0orassigneddose, not necessarilytheproposeddose.',
      contrasts=[['edits_'+str(k),'edits_1'] for k in [2,4,8]]+[['edits_4','edits_2'],['edits_8','edits_4']],
      endpoints='Ungated CA-C-N/C-N-CA angle MAE, meanreferenceCARMSD,8-outputmeanpairwiseRMSD spread; gatedCN/clashproxies, eligible/rejection fractions, finaleditcount, runtime.20kproteinbootstrapseed20261027,unadjusteddescriptive95%intervals.',
      interpretation='Report fullcurve with no automatic bestdose selection. More spread is not necessarilybettercoverage. Posthocdevelopment characterization cannot retroactivelyjustify selectingoriginal10percentcap. Newselectedmethodwouldneedfreshconfirmation. Oneproposalstream peroutput, not proof of universal causal dose-response.',length_range=[min(lengths),max(lengths)],code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(a.run/'protocol.json',protocol);print('DOSE_LOCKED',192,'outputs',doses,'callsperarm',8625,'lengths',min(lengths),max(lengths))

if __name__=='__main__':main()
