"""Snapshot a modest nonredundant experimental PDB training corpus.

RCSB 30%-cluster representatives reduce redundancy. A separate local-alignment
filter removes close matches to ALL existing benchmark sequences. This is a
documented screening heuristic, not proof of absent remote homology or ESM3
pretraining overlap. No benchmark model encoding or scoring is performed.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import requests
from Bio import Align
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def request(url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def query_candidates(output, limit):
    path = output / 'search_snapshot.json'
    if path.exists():
        return json.loads(path.read_text())['identifiers'][:limit]
    terms = [('entity_poly.rcsb_entity_polymer_type', 'exact_match', 'Protein'),
             ('entity_poly.rcsb_sample_sequence_length', 'range', {'from':40, 'to':256}),
             ('rcsb_entry_info.resolution_combined', 'less_or_equal', 3.0),
             ('rcsb_entry_info.polymer_entity_count_protein', 'equals', 1),
             ('exptl.method', 'exact_match', 'X-RAY DIFFRACTION'),
             ('rcsb_accession_info.initial_release_date', 'less_or_equal', '2025-12-31T23:59:59Z')]
    query = dict(query=dict(type='group', logical_operator='and', nodes=[dict(type='terminal', service='text',
                 parameters=dict(attribute=a, operator=o, value=v)) for a,o,v in terms]),
                 return_type='polymer_entity', request_options=dict(paginate=dict(start=0, rows=limit),
                 results_content_type=['experimental'], group_by=dict(aggregation_method='sequence_identity', similarity_cutoff=30),
                 group_by_return_type='representatives'))
    result = request('https://search.rcsb.org/rcsbsearch/v2/query', params={'json':json.dumps(query)}).json()
    ids = [r['identifier'] for r in result.get('result_set', [])]
    if not ids:
        raise RuntimeError('No PDB candidates returned')
    atomic_json(path, dict(query=query, identifiers=ids, response=result,
                          source='https://search.rcsb.org/'))
    return ids


def fetch(identifier, output):
    try:
        time.sleep(0.5)
        entry, entity = identifier.split('_')
        meta_path = output / 'metadata' / (identifier+'.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = request(f'https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}').json()
            atomic_json(meta_path, meta)
        raw = output / 'raw' / (entry+'.cif')
        if not raw.exists():
            response = request(f'https://files.rcsb.org/download/{entry}.cif', stream=True)
            tmp = raw.with_suffix('.tmp')
            size = 0
            try:
                with tmp.open('wb') as f:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 20 * 1024 * 1024:
                            raise ValueError('mmCIF exceeds pilot 20 MiB cap')
                        f.write(chunk)
                tmp.replace(raw)
            finally:
                response.close()
                tmp.unlink(missing_ok=True)
        structure = MMCIFParser(QUIET=True).get_structure(entry, str(raw))
        model = next(structure.get_models())
        full = meta['entity_poly']['pdbx_seq_one_letter_code_can'].replace('\n','').replace(' ','')
        for chain_id in meta['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']:
            if chain_id not in model:
                continue
            residues = [r for r in model[chain_id] if is_aa(r, standard=True)]
            sequence = ''.join(seq1(r.resname) for r in residues)
            if not 40 <= len(sequence) <= 256 or len(sequence) < 0.95 * len(full) or 'X' in sequence:
                continue
            if any(any(atom not in r for atom in ['N','CA','C']) for r in residues):
                continue
            backbone = np.array([[r[atom].coord for atom in ['N','CA','C']] for r in residues],dtype=np.float32)
            d = np.linalg.norm(backbone[1:,1]-backbone[:-1,1],axis=-1)
            if not np.isfinite(backbone).all() or np.any((d<2.8)|(d>4.2)):
                continue
            return dict(id=identifier, entry=entry, auth_chain=chain_id, sequence=sequence,
                        backbone=backbone, raw=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        return dict(id=identifier, error='no_complete_contiguous_chain')
    except Exception as error:
        return dict(id=identifier, error=type(error).__name__+': '+str(error)[:180])


def close_match(sequence, reference, aligner, identity=0.30, coverage=0.80):
    if sequence == reference:
        return True
    alignment = next(iter(aligner.align(sequence, reference)), None)
    if alignment is None:
        return False
    indices = alignment.indices
    valid = (indices[0] >= 0) & (indices[1] >= 0)
    a,b = indices[:,valid]
    if len(a) < coverage * min(len(sequence),len(reference)):
        return False
    matches = sum(sequence[i] == reference[j] for i,j in zip(a,b))
    return matches / max(1,len(a)) >= identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=int,default=1000)
    p.add_argument('--candidates',type=int,default=5000)
    args = p.parse_args()
    for name in ['raw','metadata','backbone']:(args.output/name).mkdir(parents=True,exist_ok=True)
    benchmark=json.loads(args.benchmark_manifest.read_text())
    references=[(t['id'],t['sequence']) for t in benchmark['all_groups']]
    aligner=Align.PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,open_gap_score=-5,extend_gap_score=-0.5)
    protocol=dict(version=2,target=args.target,candidates=args.candidates,
                  benchmark_manifest_sha256=hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest(),
                  rcsb_redundancy='30% sequence-cluster representatives',
                  screening='Local alignment >=30% identities among aligned pairs, >=80% coverage of shorter sequence; all benchmark AND already retained corpus sequences',
                  alignment=dict(match=2,mismatch=-1,open_gap=-5,extend_gap=-0.5),
                  warning='Screening heuristic, not certified remote-homology separation or absence from ESM3 pretraining. '
                          'Experimental, short, X-ray chains only; no ensemble/IDP coverage claim. Not yet latent encoded or trained.')
    existing=args.output/'protocol.json'
    if existing.exists() and json.loads(existing.read_text())!=protocol:
        raise ValueError('PDB preparation protocol mismatch')
    atomic_json(existing,protocol)
    candidates=query_candidates(args.output,args.candidates)
    retained,rejected,seen=[],[],set()
    selected_references=[]
    print('PDB_TRAINING_PREPARATION_STARTED',len(candidates),flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        iterator=iter(candidates)
        pending=deque()
        for _ in range(4):
            identifier=next(iterator,None)
            if identifier:pending.append(executor.submit(fetch,identifier,args.output))
        while pending and len(retained)<args.target:
            row=pending.popleft().result()
            identifier=next(iterator,None)
            if identifier:pending.append(executor.submit(fetch,identifier,args.output))
            if 'error' in row:
                rejected.append(dict(id=row['id'],reason=row['error']))
            else:
                sequence=row['sequence']; digest=hashlib.sha256(sequence.encode()).hexdigest()
                match=next((name for name,ref in references if close_match(sequence,ref,aligner)),None)
                corpus_match=next((name for name,ref in selected_references if close_match(sequence,ref,aligner)),None) if not match and digest not in seen else None
                if digest in seen or match or corpus_match:
                    rejected.append(dict(id=row['id'],reason='duplicate' if digest in seen else 'benchmark_sequence_match',matched=match))
                    if corpus_match:
                        rejected[-1].update(reason='within_corpus_sequence_match',matched=corpus_match)
                else:
                    np.savez_compressed(args.output/'backbone'/(row['id']+'.npz'),backbone=row.pop('backbone'))
                    row.update(sequence_hash=digest,length=len(sequence),subset='validation' if int(digest[:8],16)%10==0 else 'train',
                               backbone='backbone/'+row['id']+'.npz')
                    retained.append(row);seen.add(digest)
                    selected_references.append((row['id'],sequence))
            if (len(retained)+len(rejected))%25==0:
                atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=retained))
                atomic_json(args.output/'rejections.json',rejected)
                print(json.dumps(dict(retained=len(retained),rejected=len(rejected))),flush=True)
        for future in pending:future.cancel()
    atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=retained))
    atomic_json(args.output/'rejections.json',rejected)
    atomic_json(args.output/'summary.json',dict(protocol=protocol,retained=len(retained),rejected=len(rejected),
                rejection_counts=dict(Counter(r['reason'] for r in rejected)),
                target_reached=len(retained)>=args.target,train=sum(r['subset']=='train' for r in retained),
                validation=sum(r['subset']=='validation' for r in retained)))
    print('PDB_TRAINING_PREPARATION_COMPLETE',len(retained),flush=True)


if __name__=='__main__':main()
