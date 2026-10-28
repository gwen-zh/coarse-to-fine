"""Freeze four proposal sources; medium prediction and decoder repair are not used."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from medium_search_candidates import candidate_tables
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    patch_path=Path("artifacts/development/tokenizer_distillation_ablation/medium_kmeans512.pt");encoder_path=Path("artifacts/tokenizer/encoder_checkpoint_110.pth")
    patch=torch.load(patch_path,map_location="cpu",weights_only=False);state=torch.load(encoder_path,map_location="cpu",mmap=True,weights_only=False);state=state.get("encoder_weight",state)
    fine=state["codebook.embeddings"].numpy().copy();original=state["medium_codebook.embeddings"].numpy().copy();repaired=patch["encoder_weight"].numpy().copy();del state
    assert fine.shape==(4096,128) and repaired.shape==original.shape==(512,128)
    geometry={};arrays=dict(fine_codes=fine,repaired_medium=repaired,original_medium=original)
    for name,centers in [("repaired_cluster",repaired),("original_cluster",original)]:
        result=candidate_tables(fine,centers,4);arrays[name]=result["cluster"];arrays[name+"_fallback"]=result["fallback"];arrays[name+"_groups"]=result["group"]
        arrays["fine_neighbors"]=result["nearest"]
        sizes=result["sizes"];geometry[name]=dict(fine_centers_occupied_medium_groups=int((sizes>0).sum()),singleton_groups=int((sizes==1).sum()),largest_group=int(sizes.max()),fine_codes_requiring_fallback=int(np.sum(result["fallback"].any(-1))),nonempty_group_size_quantiles=np.quantile(sizes[sizes>0],[0,.25,.5,.75,1]).tolist(),mean_alternative_overlap_with_direct_fine_neighbors=float(np.mean([len(set(a[1:])&set(b[1:]))/4 for a,b in zip(result["cluster"],result["nearest"])])))
    np.savez_compressed(a.run/"candidates.npz",**arrays)
    parent=Path("artifacts/development/bounded_fine_search_ensemble_30s_v1");old=json.loads((parent/"protocol.json").read_text());source=Path(old["source"]);items=json.loads((source/"manifest.json").read_text())["targets"];items=sorted(items,key=lambda x:(x["sample"],x["protein"]))
    targets=[]
    for item in items:
        count=json.loads((parent/item["target"]/"summary.json").read_text())["arms"]["discrete_topk"]["forward_calls"]
        targets.append(dict(item,decoder_call_budget=count))
    protocol=dict(version=1,scope="Proposal-source development ablation:24 reused proteins x8 original coarse->fine outputs. No medium predictor, no model training, no58/64 test reuse. Only repaired medium encoder centers used, not its distilled decoder embedding table.",source=str(source),parent_search=str(parent),corpus="artifacts/pdb_training_v2",weights=old["weights"],targets=targets,input_sha256=old["input_sha256"],candidate_sha256=hashlib.sha256((a.run/"candidates.npz").read_bytes()).hexdigest(),patch_sha256=hashlib.sha256(patch_path.read_bytes()).hexdigest(),arms=["head_topk","fine_neighbors","repaired_cluster","original_cluster"],chunks=4,hardware="Allfour arms on same GPU per output, FP32 withTF32 disabled; originalCPU forward counts set budget only; noCPU/GPU token-output equivalence or speed claim",alternatives=4,delta_angstrom=1.,edit_fraction_cap=.1,candidates="Each fixed pool is anchored to originalfine token. Headtop5 versus originalfine-code Euclidean nearest4 versus nearest4 within repaired/originalmedium groups. Each pool includes origin plus4unique alternatives. If group has fewer4alternatives, fill with global nearest unused fine codes; record fallback explicitly.",group_definition="Assign each original fine code CENTER to its nearest medium center in128D encoder space. Fine Voronoi cells are not assumed nested in medium cells; same group does not guarantee decoded geometric closeness.",budget="Exactly stored30s head-search decoder forward count per output, same positions/order/gate/selection. If all violation positions vanish early, remaining calls explicitly decode current unchanged output; report padding_calls and active proposals separately. No early-stop speed claim.",gate="Relative to initial output: CN/CAcounts nonincreasing with one strict improvement,alignedCA<=1A,editedfraction<=10percent; selectlexicographic CN thenCA, rollbackoriginal ifnone. Ground truth read only after searches.",focus="Repaired_cluster minus headtopk ungated CA-C-N/C-N-CA reference angle errors, with referenceRMSD/diversity/cost safety; purefine nearest and originalmedium control separate generic neighbor and repaired-center contributions. Reused-dev descriptive inference, no confirmation.",geometry=geometry,code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path("upstream/experiments/medium_search_candidates.py"),Path("upstream/experiments/run_medium_candidate_probe.py")]})
    atomic_json(a.run/"protocol.json",protocol);print(json.dumps(geometry,indent=2),flush=True)


if __name__=="__main__":main()
