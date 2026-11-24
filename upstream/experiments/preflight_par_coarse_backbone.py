"""Real-weight transfer preflight, not a model-quality or training experiment."""
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import subprocess
import sys
import time

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_coarse_backbone import (ROOT,CHECKPOINT,GUARDED_MODULES,load_official,coarse_batch,
    attach_sequence_conditioner,ar_condition,flow_batch,sample_coarse)


def main():
    torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("Real GPU preflight requested; no CPU fallback")
    device=torch.device("cuda");capability=torch.cuda.get_device_capability()
    if capability[0]>=10:raise RuntimeError("This isolated runtime's torch2.6 build cannot execute SM100+ GPUs")
    acquisition=json.loads((ROOT/"checkpoint_acquisition.json").read_text())
    if sha(CHECKPOINT)!=acquisition["sha256"]:raise RuntimeError("Official checkpoint content changed")
    torch.manual_seed(2026)
    training_path=Path("artifacts/multiframe_early_fine_v1/train_labels.pt")
    groups=torch.load(training_path,map_location="cpu",weights_only=True)
    case=groups["PDB_train"][0]
    if case["split"]!="train":raise RuntimeError("Preflight gradient sample is not training-only")
    sequence=case["sequence"];length=len(sequence);points=(length+3)//4
    index_path=Path("artifacts/multiframe_early_fine_v1/training_feature_index.json")
    index=json.loads(index_path.read_text());feature_path=Path(index[sequence]["path"])
    with np.load(feature_path) as z:features=torch.from_numpy(z["h"].copy()).float().to(device)
    atomic_json(ROOT/"preflight_progress.json",dict(phase="loading_official",host=platform.node(),
        gpu=torch.cuda.get_device_name(),training_case=case["id"],native_development_coordinates_loaded=False))
    model,config=load_official(device)
    batch=coarse_batch(length,points,device)
    state=torch.randn((1,points,3),generator=torch.Generator().manual_seed(741)).to(device)
    state=state-state.mean(1,keepdim=True)
    with torch.no_grad():
        base_condition=model.nn(batch)["cond"]
        base_velocity=model.atom_diffusion.nn(flow_batch(batch,base_condition,state,.4))["coors_pred"]
        baseline=sample_coarse(model,None,length,813,steps=16)
    adapter=attach_sequence_conditioner(model)
    with torch.no_grad():
        new_batch,new_condition=ar_condition(model,features,length)
        new_velocity=model.atom_diffusion.nn(flow_batch(new_batch,new_condition,state,.4))["coors_pred"]
        initial=sample_coarse(model,features,length,813,steps=16)
    torch.testing.assert_close(base_condition,new_condition,atol=0,rtol=0)
    torch.testing.assert_close(base_velocity,new_velocity,atol=0,rtol=0)
    torch.testing.assert_close(baseline,initial,atol=0,rtol=0)
    params=[v for v in model.parameters() if v.requires_grad]
    frozen={name:(id(value),value._version) for name,value in model.named_parameters() if not value.requires_grad}
    buffers={name:(id(value),value._version) for name,value in model.named_buffers()}
    optimizer=torch.optim.AdamW(params,lr=1e-4)
    target=case["coarse"].to(device)[None]/10.
    target=target-target.mean(1,keepdim=True)
    time_value=.4;noise=state;corrupted=(1-time_value)*noise+time_value*target
    started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    b,z=ar_condition(model,features,length)
    predicted=model.atom_diffusion.nn(flow_batch(b,z,corrupted,time_value))["coors_pred"]
    loss=(predicted-(target-noise)).square().mean()
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite training-only transfer loss")
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm) or norm<=0:raise RuntimeError("Missing finite adapter gradient")
    optimizer.step();torch.cuda.synchronize();update_seconds=time.perf_counter()-started
    if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
        raise RuntimeError("Frozen official weights received parameter gradients")
    for name,value in model.named_parameters():
        if name in frozen and (id(value),value._version)!=frozen[name]:raise RuntimeError("Frozen official parameter changed: "+name)
    for name,value in model.named_buffers():
        if (id(value),value._version)!=buffers[name]:raise RuntimeError("Frozen official buffer changed: "+name)
    with torch.no_grad():
        _,correct=ar_condition(model,features,length)
        _,reordered=ar_condition(model,features.flip(0),length)
        response=float((correct-reordered).abs().max())
        if response<=0:raise RuntimeError("Updated sequence pathway has no order sensitivity")
        generated=sample_coarse(model,features,length,813,steps=16)
    if generated.shape!=(points,3) or not torch.isfinite(generated).all():raise RuntimeError("Invalid sampled coarse interface")
    counts=generated.new_tensor([min(4,length-4*i) for i in range(points)])
    weighted_center=(generated*counts[:,None]).sum(0)/length
    torch.testing.assert_close(weighted_center,torch.zeros(3,device=device),atol=2e-5,rtol=0)
    np.savez_compressed(ROOT/"preflight_coarse.npz",initial=initial.cpu().numpy(),
        after_discarded_update=generated.cpu().numpy())
    model.finite_preflight=True
    workspace=Path.cwd().resolve();paths=[Path(__file__),Path("upstream/experiments/par_coarse_backbone.py"),
        Path("upstream/tests/test_par_coarse_backbone.py"),Path("upstream/scripts/par_coarse_backbone_preflight.sbatch"),
        ROOT/"checkpoint_acquisition.json",CHECKPOINT,training_path,index_path,feature_path,
        Path(".venv-par-preflight/pyvenv.cfg")]
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename:
            p=Path(filename).resolve()
            if p.suffix==".py" and p.is_file() and p.is_relative_to(workspace) and not any(part.startswith('.venv') for part in p.parts):paths.append(p)
    result=dict(status="passed",checkpoint=acquisition,host=platform.node(),gpu=torch.cuda.get_device_name(),
        gpu_capability=list(capability),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        official_frozen_parameters=sum(v.numel() for v in model.parameters() if not v.requires_grad),
        trainable_sequence_parameters=sum(v.numel() for v in params),sequence_feature_dim=1536,
        block_size=4,coarse_points=points,length=length,training_case=case["id"],
        zero_initial_AR_condition_exact=True,zero_initial_velocity_exact=True,zero_initial_16step_sample_exact=True,
        finite_discarded_training_loss=float(loss.detach()),finite_adapter_gradient_norm=float(norm),
        discarded_update_seconds=update_seconds,sequence_order_stress_control_max_condition_delta=response,
        frozen_parameter_and_buffer_versions_unchanged=True,discarded_preflight_optimizer_updates=1,
        actual_trial_optimizer_updates=0,training_started=False,new_quality_claim=False,
        native_development_coordinates_loaded=False,test_coordinates_opened=False,R3_opened=False,
        runtime_compatibility=["typing-only torch.Dict alias", "restored exact nonlearned flow level-index buffer", "raise-on-use guards for config-disabled optional OpenFold blocks; no active math replacements"],
        par_source_commit=subprocess.check_output(["git","-C","third_party/par-protein","rev-parse","HEAD"],text=True).strip(),
        dependency_versions={name:importlib.metadata.version(name) for name in ["jaxtyping","loguru","torch-scatter","modelcif","ml-collections"]},
        openfold_source_commit=subprocess.check_output(["git","-C","third_party/par_preflight_openfold_v1","rev-parse","HEAD"],text=True).strip(),
        source_sha256={str(p):sha(p) for p in dict.fromkeys(paths)},
        scope="Coarse-only transfer preflight: fixed pretrained PAR modules + initially zero AA adapter. ceil(N/4) physical centers,16-step diagnostic VF solver. Not a full PAR reproduction, finetuned result, unseen-sequence or ensemble-quality claim.")
    atomic_json(ROOT/"real_weight_preflight.json",result)
    atomic_json(ROOT/"preflight_progress.json",dict(phase="complete",status="passed"))
    print("PAR_COARSE_PREFLIGHT_PASSED",result["official_frozen_parameters"],result["trainable_sequence_parameters"],update_seconds,flush=True)


if __name__=="__main__":main()
