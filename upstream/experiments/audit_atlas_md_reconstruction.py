"""Frozen tokenizer ceiling on native ATLAS train/development trajectory frames.

The sealed third trajectory is not opened. Geometry gates precede any generator
training. Native snapshots are diagnostics, not independent protein test data.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
import mdtraj as md
from atlas_md_io import backbone_indices, backbone_angstrom, uniform_frame_indices, diagnostic_trajectories
from mixed_hierarchical_generator import HybridTokenizer
from compare_two_level_ae_author import metrics
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN = Path('artifacts/atlas_md_pilot_v1')


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    selection = json.loads((RUN/'selection.json').read_text())
    download = json.loads((RUN/'download_manifest.json').read_text())
    if sha(RUN/'selection.json') != download['selection_sha256']:
        raise RuntimeError('Selection changed after download')
    for row in download['extracted']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('Changed MD input: '+row['path'])
    p = json.loads(Path('artifacts/hierarchical_ensemble_flow_v1/protocol.json').read_text())
    for path in p['tokenizer'].values():
        if sha(path) != p['source_sha256'][path]:
            raise RuntimeError('Changed frozen structural checkpoint')
    sources = [RUN/'selection.json', RUN/'download_manifest.json', Path(__file__), Path('upstream/experiments/atlas_md_io.py'), Path('upstream/tests/test_atlas_md_io.py'), Path('upstream/scripts/atlas_md_reconstruction.sbatch'), *map(Path,p['tokenizer'].values())]
    protocol_path = RUN/'reconstruction_protocol.json'
    if not protocol_path.exists():
        atomic_json(protocol_path, dict(version=1, frames_per_replicate=16, roles={'R1':'train','R2':'development','R3':'sealed_not_opened'},
            gates={'mean_ca_rmsd_A_maximum':1., 'mean_cn_outlier_rate_maximum':.05, 'mean_boundary_cn_rate_maximum':.15},
            gate_scope='Engineering check of frozen-decoder reconstruction ceiling, not a test of generation or thermodynamic fidelity. Both inspected replicates must pass before a native-MD flow fit.',
            coordinate_units='MDTraj loads nm; multiply by 10 exactly once to obtain Angstrom before encoding.',
            frame_selection='16 evenly spaced indices including trajectory endpoints; no structure-quality filtering or frame cherry-picking.',
            tokenizer=p['tokenizer'], source_sha256={str(path):sha(path) for path in sources}))
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed reconstruction protocol source '+path)
    out = RUN/'reconstruction'
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('Frozen reconstruction diagnostic is already complete')
    pdbs = list((RUN/'raw').rglob('*.pdb'))
    if len(pdbs) != 1:
        raise RuntimeError('Expected a single explicit protein topology')
    topology = md.load_topology(str(pdbs[0]))
    indices = backbone_indices(topology, selection['selected']['sequence'])
    decoder = HybridTokenizer.from_checkpoints(**protocol['tokenizer'])
    groups = []
    all_rows = []
    for replicate, path in diagnostic_trajectories(RUN/'raw').items():
        with md.open(str(path)) as trajectory:
            total = len(trajectory)
        frames = uniform_frame_indices(total, protocol['frames_per_replicate'])
        rows = []
        references, reconstructions, labels_c, labels_f, tokens = [], [], [], [], []
        times = []
        for index in frames:
            frame = md.load_frame(str(path), int(index), top=topology)
            bb = backbone_angstrom(frame.xyz, indices)[0]
            original = metrics(bb, bb)
            # Fail on broken/wrapped input; do not silently transform the MD source.
            if original['cn_rate'] > .01:
                raise RuntimeError('Native MD has nonphysical or wrapped backbone links')
            with torch.no_grad():
                c, f, target = decoder.model.encode_latents(torch.from_numpy(bb))
                ids = decoder.model.fine_book.nearest(f)
                rec = decoder.decode(c, ids).numpy()
            row = dict(replicate=replicate, frame=int(index), time_ps=float(frame.time[0]), reconstruction=metrics(rec, bb), reference_geometry=original)
            rows.append(row)
            references.append(bb);reconstructions.append(rec);labels_c.append(c.numpy());labels_f.append(f.numpy());tokens.append(ids.numpy());times.append(float(frame.time[0]))
        averaged = {key:float(np.mean([r['reconstruction'][key] for r in rows])) for key in rows[0]['reconstruction']}
        gate = dict(mean_ca_rmsd_A_at_most1=averaged['ca_rmsd']<=1., mean_cn_rate_at_most5pct=averaged['cn_rate']<=.05, mean_boundary_cn_rate_at_most15pct=averaged['boundary_cn_rate']<=.15)
        result = dict(replicate=replicate, role=protocol['roles'][replicate], total_native_frames=total, inspected_frames=frames.tolist(), first_inspected_time_ps=times[0], last_inspected_time_ps=times[-1], metrics=averaged, gate=gate, pass_gate=all(gate.values()))
        groups.append(result);all_rows.extend(rows)
        np.savez_compressed(out/f'{replicate}_diagnostic.npz', backbone=np.stack(references), reconstruction=np.stack(reconstructions), coarse=np.stack(labels_c), fine_continuous=np.stack(labels_f), fine_ids=np.stack(tokens), frame_indices=frames, time_ps=np.array(times), sequence=np.array(selection['selected']['sequence']))
        atomic_json(out/f'{replicate}.json',dict(summary=result,frames=rows))
        print('MD_RECONSTRUCTION_REPLICATE',json.dumps(result),flush=True)
    result = dict(status='complete', name=selection['selected']['name'], residues=len(indices), replicates=groups, frames=all_rows, pass_both_reconstruction_gates=all(r['pass_gate'] for r in groups), sealed_test_opened=False, generator_training_started=False, seconds=time.perf_counter()-started, protocol_sha256=sha(protocol_path))
    atomic_json(out/'complete.json',result)
    print('MD_RECONSTRUCTION_COMPLETE',result['pass_both_reconstruction_gates'],flush=True)


if __name__=='__main__':main()
