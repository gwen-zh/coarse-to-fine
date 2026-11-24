"""Numerical check of exact geometry gradients through the continuous decoder.

No STE, optimization, weight updates or validation examples. Fixed case indices
0,4,32,36 are the first two PDB and first two MD training proteins in the frozen
diagnostic ordering. This is a gradient check, not a performance experiment.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnostic_io import atomic_json
from geometry_objective import backbone_penalties
from hierarchical_pipeline_io import sha
from run_latent_source_diagnostic import verify, RUN
from run_sequence_multiscale_ensemble import load_codec


def geometry_loss(codec,coarse,fine):
    # The convenience codec.decode is decorated no_grad for inference. Call the
    # same frozen fine decoder directly to retain gradients with respect to fine.
    bb=codec.fine_tokenizer.decode_vectors(fine,codec.context(coarse,len(fine)),len(fine))
    cn,clash=backbone_penalties(bb)
    distance=(bb[:-1,2]-bb[1:,0]).norm(dim=-1)
    ids=torch.arange(len(bb)-1)
    boundary=ids//4!=(ids+1)//4
    boundary_loss=(F.relu(1.1-distance[boundary]).square()+F.relu(distance[boundary]-1.7).square()).mean()
    return cn+boundary_loss+.1*clash,bb


def main():
    torch.set_num_threads(2)
    p=json.loads((RUN/"protocol.json").read_text());verify(p)
    selection=json.loads((RUN/"seed2026"/"complete.json").read_text())["records"]
    indices=[0,4,32,36]
    codec=load_codec(p,"centers3");before={k:v.clone() for k,v in codec.state_dict().items()}
    rows=[];inputs=[RUN/"protocol.json",Path(__file__),Path("upstream/experiments/geometry_objective.py")]
    for seed in [2026,2027]:
        for index in indices:
            path=RUN/f"seed{seed}"/f"case_{index:03d}.npz";inputs.append(path)
            with np.load(path,allow_pickle=False) as data:
                coarse=torch.from_numpy(data["native_coarse"].copy())
                fine=torch.from_numpy(data["oracle_fine16"].copy()).requires_grad_(True)
                baseline=torch.from_numpy(data["true_coarse_generated_fine16"].copy())
            loss,bb=geometry_loss(codec,coarse,fine)
            # Autograd-enabled Linear kernels can differ by a few float32 ULPs
            # from no-grad inference. Use the established codec parity tolerance.
            torch.testing.assert_close(bb,baseline,rtol=2e-5,atol=2e-5)
            forward_error=float((bb.detach()-baseline).abs().max())
            gradient=torch.autograd.grad(loss,fine)[0]
            if not torch.isfinite(gradient).all() or gradient.norm()<=0:
                raise RuntimeError("Geometry gradient is absent or nonfinite")
            direction=gradient/gradient.norm()
            epsilon=1e-3
            with torch.no_grad():
                plus,_=geometry_loss(codec,coarse,fine+epsilon*direction)
                minus,_=geometry_loss(codec,coarse,fine-epsilon*direction)
                finite_difference=(plus-minus)/(2*epsilon)
                analytic=(gradient*direction).sum()
                relative=float((finite_difference-analytic).abs()/torch.maximum(
                    finite_difference.abs(),analytic.abs()).clamp_min(1e-6))
                if relative>.05:raise RuntimeError("Finite difference disagrees with decoder gradient")
            rows.append(dict(seed=seed,case_id=selection[index]["case_id"],
                domain=selection[index]["domain"],geometry_proxy=float(loss.detach()),
                fine_gradient_norm=float(gradient.norm()),directional_finite_difference=float(finite_difference),
                directional_autograd=float(analytic),relative_error=relative,
                production_forward_numeric_parity=True,forward_max_abs_error_A=forward_error))
    for k,v in codec.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    atomic_json(RUN/"continuous_geometry_gradient_audit.json",dict(status="passed",rows=rows,
        source_sha256={str(path):sha(path) for path in inputs},epsilon=1e-3,
        forward_parity_atol=2e-5,forward_parity_rtol=2e-5,
        method="Exact continuous fine input gradients; no codebook snap or STE.",
        decoder_weights_unchanged=True,optimization_performed=False,
        scope="Eight fixed training-case/seed checks. Establishes a differentiable training interface only; no quality, generalization or ensemble-preservation claim.",
        next="A separately controlled fine-flow versus fine-flow-plus-geometry training pilot can use this interface. The convenience inference wrapper must not be used in the gradient path."))
    print("CONTINUOUS_FINE_EXACT_GRADIENT_CHECK_PASSED",len(rows),"cases",flush=True)


if __name__=="__main__":main()
