"""Posthoc frame-only stability analysis; no second tokenizer fit or R3 reuse."""
from pathlib import Path
import json,numpy as np,torch
from block_transport_frames import transported_block_frames
from block_local_frames import vectors_to_local
from hierarchical_latent_contract import remove_block_translation
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/block_frame_tokenizer_v1')
OUT=Path('artifacts/block_transport_geometry_v1')


def stats(frames,residual):
    relative=frames[:-1].transpose(-1,-2)@frames[1:];angle=torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))).numpy()
    return dict(frame_step_angle_median_degrees=float(np.median(angle)),frame_step_angle_p90_degrees=float(np.percentile(angle,90)),frame_step_angle_p99_degrees=float(np.percentile(angle,99)),frame_step_angle_over90deg_fraction=float((angle>90).mean()),residual_coordinate_variance=float(residual.var(0,unbiased=False).mean()))


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():raise RuntimeError('Frame-only analysis already completed')
    atomic_json(OUT/'protocol.json',dict(scope='Posthoc geometry-only follow-up, not a new trained model. Same R1 training frames; no R2 selection or R3 reuse. Original v1 fine-model gates unchanged.',source_sha256={str(x):sha(x) for x in [Path(__file__),Path('upstream/experiments/block_transport_frames.py'),Path('upstream/tests/test_block_transport_frames.py'),RUN/'labels.pt',RUN/'report.json']}))
    data=torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True);cases=data['MD_train'];old_frames=torch.stack([r['contexts']['local']['basis'] for r in cases]);new_frames=[];halfturns=[]
    residual=torch.stack([remove_block_translation(r['target']) for r in cases]);local_old=torch.stack([vectors_to_local(x,f) for x,f in zip(residual,old_frames)])
    for case in cases:
        frames,meta=transported_block_frames(case['contexts']['local']['centers']);new_frames.append(frames);halfturns.append(meta['near_antiparallel_transports'])
    new_frames=torch.stack(new_frames);local_new=torch.stack([vectors_to_local(x,f) for x,f in zip(residual,new_frames)])
    result=dict(status='complete',train_frames=len(cases),tests_passed=4,original_frame=stats(old_frames,local_old),transported_frame=stats(new_frames,local_new),global_reference_residual_variance=float(residual.var(0,unbiased=False).mean()),near_antiparallel_transports=sum(halfturns),new_tokenizer_trained=False,original_models_modified=False,R3_loaded=False,scope='Geometry stability and target-variance descriptors only. They do not prove better reconstruction or generation. New frame has not been adopted.')
    atomic_json(OUT/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
