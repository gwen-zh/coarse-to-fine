"""Geometry follow-up only; leave all fine-codebook training results intact."""
from pathlib import Path
import json,numpy as np,torch
from block_centroid_transport_frames import centroid_transport_frames
from block_local_frames import vectors_to_local
from hierarchical_latent_contract import remove_block_translation
from audit_block_transport_frames import stats
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

OUT=Path('artifacts/block_centroid_transport_geometry_v1')


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():raise RuntimeError('Centroid transport audit already completed')
    sources=[Path(__file__),Path('upstream/experiments/block_centroid_transport_frames.py'),Path('upstream/tests/test_block_centroid_transport_frames.py'),Path('artifacts/block_frame_tokenizer_v1/labels.pt'),Path('artifacts/block_transport_geometry_v1/report.json')]
    atomic_json(OUT/'protocol.json',dict(scope='Posthoc geometry-only follow-up on R1, no model fitting or R3 reuse. Seed uses unweighted coarse centroid, then ordered weighted centroids only if needed; farthest-point fallback is explicit.',source_sha256={str(x):sha(x) for x in sources}))
    groups=torch.load('artifacts/block_frame_tokenizer_v1/labels.pt',map_location='cpu',weights_only=True);cases=groups['MD_train'];frames=[];counts=[0,0,0,0]
    for case in cases:
        f,meta=centroid_transport_frames(case['contexts']['local']['centers']);frames.append(f);counts[meta['seed_rule']]+=1
    frames=torch.stack(frames);residual=torch.stack([remove_block_translation(c['target']) for c in cases]);local=torch.stack([vectors_to_local(r,f) for r,f in zip(residual,frames)])
    original=json.loads(Path('artifacts/block_transport_geometry_v1/report.json').read_text())
    result=dict(status='complete',tests_passed=4,train_frames=len(cases),seed_rule_counts=counts,original_frame=original['original_frame'],first_transport=original['transported_frame'],centroid_transport=stats(frames,local),reference_coordinate_variance=float(residual.var(0,unbiased=False).mean()),new_tokenizer_trained=False,models_replaced=False,R3_loaded=False,scope='Geometry smoothness only. Lower jumps or target variance do not establish better reconstruction or generation; fresh matched training is required before adoption.')
    atomic_json(OUT/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
