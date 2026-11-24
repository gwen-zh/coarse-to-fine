"""Post-hoc input-range diagnostic, not a causal attribution or model selector."""
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha


ROOT=Path("artifacts/decoder_angle_adaptation_v1")
TRAIN=Path("artifacts/multiframe_early_fine_v1/train_labels.pt")


def describe(values):
    values=np.asarray(values,float)
    outside=np.abs(values)>1.+1e-6
    return dict(residues=len(values),outside_encoder_tanh_range_component_fraction=float(outside.mean()),
        outside_encoder_tanh_range_residue_fraction=float(outside.any(-1).mean()),
        absolute_value_99th_percentile=float(np.quantile(np.abs(values),.99)),
        maximum_absolute_value=float(np.abs(values).max()),
        per_dimension_mean=values.mean(0).tolist(),per_dimension_std=values.std(0).tolist())


def main():
    torch.set_num_threads(2)
    inputs=ROOT/"evaluation_inputs.pt"
    groups=torch.load(inputs,map_location="cpu",weights_only=True)
    train=torch.load(TRAIN,map_location="cpu",weights_only=True)
    rows=[dict(seed=g["seed"],protein=g["protein"],outlet=g["outlet"],
        **describe(g["fine"].numpy().reshape(-1,8))) for g in groups]
    summary={outlet:describe(np.concatenate([g["fine"].numpy().reshape(-1,8) for g in groups if g["outlet"]==outlet]))
        for outlet in ["author_roundtrip","own_frozen_generator"]}
    native={"PDB_train":describe(torch.cat([c["fine_continuous"] for c in train["PDB_train"]]).numpy()),
        "MD_train":describe(torch.cat([c["fine_continuous"] for cases in train["train"].values() for c in cases]).numpy())}
    report=dict(status="complete_posthoc_input_range_diagnostic",rows=rows,summary=summary,native_training_latents=native,
        scope="The frozen encoder has a tanh output, so values outside[-1,1] cannot be exact encoder outputs. Range overlap is not manifold membership; this cannot isolate the cause of poor decoded geometry or prove clipping helps. No model update, clipping, extra sample or development reference coordinate is used.",
        source_sha256={str(p):sha(p) for p in [Path(__file__),inputs,TRAIN]})
    atomic_json(ROOT/"latent_input_range_audit.json",report)
    print({key:{k:v for k,v in value.items() if k not in ["per_dimension_mean","per_dimension_std"]} for key,value in summary.items()})


if __name__=="__main__":main()
