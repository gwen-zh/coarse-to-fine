"""Diagnostic-only 2x2: reconstructed/exact centers x snapped/continuous fine.

Exact-center arms read native coordinates and are explicitly oracle diagnostics,
not deployable methods. No weight fitting, new MD frames, or R3 access.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from two_level_autoencoder import pack
from hierarchical_latent_contract import pool_ca, compose_levels
from hierarchical_rigid_vq import canonical_backbone, rigid_backbone
from mixed_hierarchical_generator import HybridTokenizer
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN = Path('artifacts/atlas_md_pilot_v1')
ARMS = ['reconstructed_centers_snapped_fine', 'reconstructed_centers_continuous_fine', 'exact_centers_snapped_fine', 'exact_centers_continuous_fine']


def decode_with_centers(model, centers, fine, length):
    packed, valid = pack(fine, model.block_size)
    previous = F.pad(centers[1:]-centers[:-1], (0,0,1,0))/10.
    following = F.pad(centers[1:]-centers[:-1], (0,0,0,1))/10.
    blocks = torch.cat([packed.flatten(1), valid.to(fine.dtype)], -1)
    inputs = torch.cat([packed.flatten(1), previous, following, valid.to(fine.dtype), F.pad(blocks[:-1],(0,0,1,0)), F.pad(blocks[1:],(0,0,0,1))], -1)
    raw = model.fine_decoder(inputs).reshape(-1,9)[:length]
    return compose_levels(centers, rigid_backbone(raw), model.block_size)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    p = json.loads((RUN/'reconstruction_protocol.json').read_text())
    for path, digest in p['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed native-MD reconstruction source '+path)
    sources = [Path(__file__), RUN/'reconstruction/complete.json', RUN/'reconstruction_protocol.json', *[RUN/f'reconstruction/{r}_diagnostic.npz' for r in ['R1','R2']]]
    protocol_path = RUN/'decomposition_protocol.json'
    if protocol_path.exists():
        raise RuntimeError('Use a new diagnostic namespace')
    atomic_json(protocol_path, dict(version=1, arms=ARMS, frames='Same 16 predetermined frames from R1/R2; no frame selection or R3 loading', oracle_arms=['exact_centers_snapped_fine','exact_centers_continuous_fine'], scope='Oracle factor isolation only, not deployable generation or a trained repair', source_sha256={str(x):sha(x) for x in sources}))
    decoder = HybridTokenizer.from_checkpoints(**p['tokenizer'])
    rows=[]
    parity=0.
    for replicate in ['R1','R2']:
        with np.load(RUN/f'reconstruction/{replicate}_diagnostic.npz', allow_pickle=False) as z:
            references=z['backbone'].copy();old=z['reconstruction'].copy();c=z['coarse'].copy();f=z['fine_continuous'].copy();ids=z['fine_ids'].copy();indices=z['frame_indices'].copy()
        for index, reference in enumerate(references):
            coarse=torch.from_numpy(c[index]);fine=torch.from_numpy(f[index]);quantized=decoder.model.fine_book.codes[torch.from_numpy(ids[index])]
            target=canonical_backbone(torch.from_numpy(reference));exact=pool_ca(target,4)
            centers=decoder.coarse_centers(coarse,len(reference))
            for arm in ARMS:
                anchor=exact if arm.startswith('exact_') else centers
                fine_input=fine if arm.endswith('continuous_fine') else quantized
                prediction=decode_with_centers(decoder.model,anchor,fine_input,len(reference)).numpy()
                if arm==ARMS[0]:
                    difference=float(np.max(np.abs(prediction-old[index])));parity=max(parity,difference)
                    np.testing.assert_allclose(prediction,old[index],atol=2e-5,rtol=2e-5)
                rows.append(dict(replicate=replicate,frame=int(indices[index]),arm=arm,metrics=metrics(prediction,reference)))
    groups=[]
    for replicate in ['R1','R2']:
        for arm in ARMS:
            subset=[r for r in rows if r['replicate']==replicate and r['arm']==arm]
            groups.append(dict(replicate=replicate,arm=arm,frames=len(subset),metrics={k:float(np.mean([r['metrics'][k] for r in subset])) for k in subset[0]['metrics']}))
    result=dict(status='complete',source_protocol_sha256=sha(protocol_path),maximum_original_decode_parity_error=parity,groups=groups,frames=rows,weights_updated=False,sealed_test_opened=False,generator_training_started=False)
    atomic_json(RUN/'reconstruction_decomposition.json',result)
    for group in groups:
        print(json.dumps(dict(replicate=group['replicate'],arm=group['arm'],**{key:group['metrics'][key] for key in ['ca_rmsd','cn_rate','boundary_cn_rate']})),flush=True)


if __name__=='__main__':main()
