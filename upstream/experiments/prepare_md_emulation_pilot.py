"""Outcome-blind pilot inventory and official numerical-interface check."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from Bio import Align
from prepare_pdb_training import close_match
from diagnostic_io import atomic_json
from md_projection_adapter import ca_trajectory

ROOT=Path('upstream/bioemu-benchmarks')
sys.path.insert(0,str(ROOT.resolve()))
from bioemu_benchmarks.eval.md_emulation.projections import (compute_features,FeatureSettings,ProjectionParameters,project_features)
import mdtraj as md


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=Path('artifacts/md_emulation_pilot_v1');out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'bioemu_benchmarks/assets/md_emulation_benchmark_0.1/md_emulation'
    sources=[Path('artifacts/pdb_training_v2/manifest.json'),Path('artifacts/development/manifest.json'),Path('artifacts/fine_search_confirmation_v1/manifest.json')]
    reference=[]
    for p in sources:
        data=json.loads(p.read_text())
        reference.extend((str(p),x['id'],x['sequence']) for x in data.get('all_groups',data.get('targets',[])))
    rows=list(csv.DictReader((base/'testcases.csv').open()))
    aligner=Align.PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,open_gap_score=-5,extend_gap_score=-.5)
    audited=[]
    with np.load(base/'reference_projections.npz') as refs,np.load(base/'projections_mean.npz') as means,np.load(base/'projections_sqrt_inv_cov.npz') as matrices:
        for row in rows:
            name=row['test_case'];n=len(row['sequence']);features=(n-4)*(n-3)//2
            if refs[name].shape[1]!=2 or means[name].shape!=(features,) or matrices[name].shape!=(features,2):raise RuntimeError('Projection dimension mismatch '+name)
            matches=[dict(source=p,id=identifier,exact=sequence==row['sequence']) for p,identifier,sequence in reference if close_match(row['sequence'],sequence,aligner)]
            audited.append(dict(id=name,sequence=row['sequence'],length=n,reference_projection_rows=len(refs[name]),overlap_matches=matches))
        # Official full-atom fixture -> our CA-only adapter -> official projection.
        fixture='cath1_1bl0A02';directory=ROOT/'tests/test_data/samples_example/md_emulation'
        trajectory=md.load(str(directory/(fixture+'.xtc')),top=str(directory/(fixture+'.pdb')))
        seq=next(x['sequence'] for x in audited if x['id']==fixture)
        ca=trajectory.xyz[:,trajectory.top.select('name CA')]*10
        bridge=ca_trajectory(seq,ca)
        expected_features=np.load(ROOT/f'tests/test_data/md_emulation/test_{fixture}_features.npy')
        expected_projection=np.load(ROOT/f'tests/test_data/md_emulation/test_{fixture}_projections.npy')
        actual=compute_features(bridge,FeatureSettings())
        projected=project_features(actual,ProjectionParameters(mean=means[fixture],sqrt_inv_cov=matrices[fixture]))
        np.testing.assert_allclose(actual,expected_features,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(projected,expected_projection,rtol=1e-5,atol=1e-5)
        smoke=dict(samples=len(ca),features_max_abs=float(abs(actual-expected_features).max()),projections_max_abs=float(abs(projected-expected_projection).max()),passed=True)
    eligible=sorted((r for r in audited if not r['overlap_matches']),key=lambda r:hashlib.sha256(('md-pilot-v1:'+r['id']).encode()).hexdigest())
    pilot=eligible[:5]
    protocol=dict(commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        benchmark='BioEmu MD emulation v0.1',source='https://github.com/microsoft/bioemu-benchmarks',
        stage='Inventory and numeric compatibility; no generated samples evaluated',
        targets=pilot,selection='First five SHA256(md-pilot-v1:<id>) ordered targets passing project sequence screen, before model outcomes',
        screening='Local alignment >=30% identity and >=80% shorter-sequence coverage; heuristic, not certified homology or pretraining isolation',
        planned_comparison=['original frozen path','original plus existing frozen constrained fine-token search'],
        planned_sample_counts=[32,128,512],seeds=[2026,2027,2028],
        policy='Nested sample-count diagnostics; paired original/search samples; no MD-score hyperparameter selection. Report raw invalid fraction separately. No silent filtering or duplicating output to inflate sample size.',
        reference_scope='17 systems with precomputed 2D reference projections. No production raw MD trajectories or per-trajectory/time/replica provenance provided by these assets; cannot audit MD convergence or kinetics.',
        metric_scope='Official projected free-energy MAE/RMSE/coverage with official settings, plus raw geometry/spread. CA-only adapter supports projection, not all-atom validity filters. Resampling for density smoothing is not extra independent model samples.',
        sha256={str(p):sha(p) for p in sources+sorted(base.glob('*')) if p.is_file()})
    atomic_json(out/'inventory.json',dict(targets=audited,screened_reference_sequences=len(reference),eligible=len(eligible),fixture_check=smoke))
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Pilot protocol differs')
    atomic_json(pp,protocol)
    print(json.dumps(dict(total=len(rows),eligible=len(eligible),pilot=[x['id'] for x in pilot],fixture_check=smoke)))

if __name__=='__main__':main()
