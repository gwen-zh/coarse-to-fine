"""Freeze a no-training native-coarse sampler experiment on reused development data."""
import argparse,hashlib,json
from pathlib import Path
from diagnostic_io import atomic_json


def fingerprint(path):
    path=Path(path);h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    st=path.stat();return dict(path=str(path),sha256=h.hexdigest(),size=st.st_size,mtime_ns=st.st_mtime_ns)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    corpus=Path("artifacts/pdb_training_v2");names=json.loads(Path("artifacts/development/objective_geometry_diagnosis/protocol.json").read_text())["geometry_targets"];rows={x["id"]:x for x in json.loads((corpus/"manifest.json").read_text())["targets"]};targets=[rows[name] for name in names];assert len(targets)==24 and all(x["subset"]=="validation" for x in targets)
    generator=fingerprint("artifacts/mp_rank_00_model_states.pt");decoder=fingerprint("artifacts/tokenizer/decoder_checkpoint_110.pth")
    source_files=["upstream/experiments/discrete_flow_math.py","upstream/experiments/discrete_flow_sampler.py","upstream/experiments/sample_discrete_flow_probe.py","upstream/hierarchical_ConfGen/slm/models/model.py","upstream/hierarchical_ConfGen/slm/utils/noise_utils.py","upstream/hierarchical_ConfGen/slm/utils/checkpoint_utils.py","upstream/hierarchical_ConfGen/configs/experiment/mdlm.yaml","upstream/hierarchical_ConfGen/configs/model/default.yaml"]
    cfg=Path(".hydra/config.yaml")
    if cfg.exists():source_files.append(str(cfg))
    protocol=dict(version=1,scope="No-training coarse corrector inference experiment on24 reused development proteins x8 samples. Original32-token semantics, pretrained backbone, fine argmax and decoder frozen. No LoRA or geometry/reference-based selection. No58/64 test use.",corpus=str(corpus),targets=targets,chunks=4,decode_chunks=8,samples=8,coarse_steps=[5,20],eps=1e-5,arms=[dict(name="native",beta=None),dict(name="base_fm",beta=0.),dict(name="corrector_b0p5",beta=.5),dict(name="corrector_b1",beta=1.)],primary="Mean per-protein8sample reference CA RMSD: corrector beta0.5 at20 coarse updates minus native20; reused-development inference only",secondary="All arms/budgets reference error, CN and CA geometry, ungated bond angles,8sample spread/best reference match, actual warm GPU generation time, exact coarse/fine model calls, observed-grid remask/revision counts",sampler="kappa=1-(1-exp(-sigma(native_diffusion_time))). Corrector(1+beta)*forward-beta*backward with mask-source prior. A=(1+beta)log(m_t/m_s),B=beta*log((1-m_s)/(1-m_t)); bounded two-status frozen-hazard local exponential. Beta0 uses identical native raw weights/RNG operation order.",limits="For beta>0 integrator is a finite-step first-order approximation, not exact inhomogeneous joint CTMC. Perfect-posterior ideal marginal guarantee does not hold automatically for this learned checkpoint. Source/end epsilon and final native argmax retained. Same NFE not identical total FLOPs/runtime. Tracing overhead included.",calls="Every arm atK updates requires exactlyK+1 coarse plus1 fine backbone forward;5updates=7calls,20updates=22calls. No hidden posterior calls in corrector.",equivalence="Preflight2proteins x2seeds x2budgets native/base exact coarse and fine token replay; all192 full cases also verify native/base arrays equal for both budgets",source_code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},generator=generator,decoder=decoder,failure_policy="All24x8x4x2 outputs required; no output exclusion or references in sampling; upstream failure blocks downstream. Checkpoint/config/seed/beta/budget fixed before results.")
    atomic_json(a.run/"protocol.json",protocol);print("DISCRETE_FLOW_PROTOCOL_FROZEN",len(targets),flush=True)


if __name__=="__main__":main()
