"""Freeze the ordered coarse geometry study and its phase gates."""
import ast,json
from pathlib import Path
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from coarse_geometry_supervision import ARMS
RUN=Path('artifacts/coarse_geometry_supervision_v1')

def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Geometry protocol already frozen')
    parent=Path('artifacts/coarse_generalization_v1/protocol.json');old=json.loads(parent.read_text());p=dict(old);p['source_sha256']=dict(old['source_sha256'])
    p.update(arms=ARMS,small_train_ids=old['train_probe_ids'],lr=3e-5,min_lr=3e-6,
        small=dict(updates=1024,checkpoints=[128,512,1024],required_training_improvement=.25),
        full=dict(updates=899*12,checkpoints=[899,899*3,899*6,899*12]),unit_tests_passed=7,
        scope='Fixedrepresentation,coarseonly,newgeometryobjectives. Two seeds,threearms;small8trainfit→gatednew899fulltraining. Initialweightsoriginal60epoch,notfailedcontinuationorsmallfit. Learningrate3e-5→3e-6sharedwithnewcontrol;notdirectcomparisonofobjectivetoearlierhighLRruns. Development16reused,notconfirmation;noencoder/decoder/fineupdates,noactualreferenceacceptance,58/64testsunused. Finephasehelduntilcoarselayoutengineeringgate;decodersearchcombinationnotvalidated.')
    files=[parent,Path('artifacts/coarse_generalization_v1/review.json')]
    for name in ['coarse_geometry_supervision.py','run_coarse_geometry.py','prepare_coarse_geometry.py','run_coarse_generalization.py','run_coarse_smallset.py']:
        path=Path('upstream/experiments')/name;ast.parse(path.read_text());files.append(path)
    files += [Path('upstream/tests/test_coarse_geometry_supervision.py'),Path('upstream/scripts/coarse_geometry.sbatch'),Path('paper/protocols/coarse_geometry_supervision_zh.rst')]
    p['source_sha256'].update({str(x):sha(x) for x in files})
    atomic_json(RUN/'protocol.json',p);print('COARSE_GEOMETRY_PROTOCOL_FROZEN',len(p['source_sha256']),flush=True)
if __name__=='__main__':main()
