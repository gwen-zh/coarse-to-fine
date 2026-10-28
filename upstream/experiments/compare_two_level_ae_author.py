"""Same-input reconstruction comparison; not matched rate or sequence generation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from structural_metrics import aligned_rmsd, summarize_backbone
from backbone_quality_audit import measurements

RUN = Path('artifacts/development/two_level_ae_author_comparison_v1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def validate_pair(pred, reference, length):
    expected = (length, 3, 3)
    if pred.shape != expected or reference.shape != expected:
        raise ValueError('Expected exact same unpadded N/CA/C chain: '+str(expected))
    if not np.isfinite(pred).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite chain; do not exclude residues or proteins')


def metrics(pred, reference):
    validate_pair(pred, reference, len(reference))
    pred, reference = np.asarray(pred, float), np.asarray(reference, float)
    m = summarize_backbone(pred, [reference[:, 1]])
    ids = np.arange(len(pred)-1); cross = ids//4 != (ids+1)//4
    distance = np.linalg.norm(pred[:-1, 2]-pred[1:, 0], axis=-1)
    bad = (distance<1.) | (distance>1.8)
    if not cross.any() or cross.all(): raise ValueError('Comparison requires both within- and cross-block bonds')
    return dict(ca_rmsd=aligned_rmsd(pred[:, 1], reference[:, 1]),
                backbone_rmsd=aligned_rmsd(pred.reshape(-1, 3), reference.reshape(-1, 3)),
                cn_rate=m['backbone_cn_outliers']/(len(pred)-1),
                within_cn_rate=float(bad[~cross].mean()), boundary_cn_rate=float(bad[cross].mean()),
                ca_clashes_per_residue=m['upstream_ca_clashes']/len(pred),
                ca_clash_valid=float(m['upstream_ca_clash_valid']),
                **measurements(pred, reference))


def paired_rows(rows, targets):
    expected = {r['id']: r['subset'] for r in targets}
    if len(rows)!=len(expected) or {r['target']:r['subset'] for r in rows}!=expected:
        raise ValueError('Incomplete or duplicate protein pairing; do not average a subset')
    return rows


def verify(p):
    for source, digest in p['sha256'].items():
        if sha(source)!=digest: raise RuntimeError('Frozen input changed: '+source)


def author(p):
    import sys
    import torch
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'Hierarchical_VQ_VAE'))
    from esm.models.new_vqvae import ProjectedEncoder, ProjectedDecoder, apply_special_tokens
    from evaluate_residual_vq import decode_embedding
    torch.set_num_threads(p['threads']); torch.manual_seed(20261027)
    encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096)
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    for model, path, key in [(encoder,p['encoder_weights'],'encoder_weight'),(decoder,p['decoder_weights'],'decode_proj')]:
        weights = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(weights.get(key, weights), strict=True)
        model.eval().requires_grad_(False)
        del weights
    directory = RUN/'author'; directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists(): raise RuntimeError('Author comparison already complete')
    counts = dict(encoder=sum(v.numel() for v in encoder.parameters()), decoder=sum(v.numel() for v in decoder.parameters()))
    rows = []; parity = None
    with torch.no_grad():
        for item in p['targets']:
            with np.load(item['reference']) as data: x = torch.from_numpy(data['backbone'].copy())
            with np.load(item['cached_fine']) as data: cached = data['fine'].copy()
            validate_pair(x.numpy(), x.numpy(), item['length'])
            started = time.perf_counter(); encoded = encoder.encode(x[None]); fine = encoded[3][0]
            if fine.shape!=(item['length'],) or fine.min()<0 or fine.max()>=4096: raise RuntimeError('Invalid original fine codes')
            pred = decode_embedding(decoder, decoder.embed(fine))
            seconds = time.perf_counter()-started
            if parity is None:
                lengths = torch.tensor([len(x)])
                padded = [apply_special_tokens(v, lengths, k+2, k+1, k+3) for v,k in zip(encoded[3:6],[4096,512,32])]
                full = decoder.decode(*padded, lengths)['bb_pred_large'][0, 1:len(x)+1]
                torch.testing.assert_close(pred, full, atol=2e-5, rtol=2e-5)
                parity = dict(target=item['id'], max_abs_error=float((pred-full).abs().max()),
                              fine_only_matches_original_full_decode=True, absolute_tolerance=2e-5, relative_tolerance=2e-5)
                atomic_json(directory/'interface_parity.json', parity)
            prediction = pred.numpy(); fine_ids = fine.numpy()
            row = dict(target=item['id'], subset=item['subset'], length=len(x),
                       cached_fine_differences=int((fine_ids!=cached).sum()), encode_decode_seconds=seconds,
                       metrics=metrics(prediction,x.numpy()), reference_metrics=metrics(x.numpy(),x.numpy()))
            np.savez_compressed(directory/(item['id']+'.npz'), backbone=prediction, fine=fine_ids)
            atomic_json(directory/(item['id']+'.json'),row); rows.append(row)
            atomic_json(directory/'progress.json',dict(completed=len(rows),total=len(p['targets']),last_target=item['id']))
            print('AUTHOR_AE_RECONSTRUCTION',json.dumps(row),flush=True)
    paired_rows(rows,p['targets'])
    atomic_json(directory/'complete.json',dict(status='complete',targets=rows,parameters=counts,interface=parity,
                cached_fine_total_differences=sum(r['cached_fine_differences'] for r in rows),
                policy='Primary original encoder freshly reads exact reference NPZ; cache differences are reported, never used to select the better output.'))
    print('AUTHOR_AE_RECONSTRUCTION_COMPLETE',flush=True)


def report(p):
    author_data = json.loads((RUN/'author/complete.json').read_text())
    if author_data['status']!='complete': raise RuntimeError('Original reconstruction incomplete')
    original = {r['target']:r for r in paired_rows(author_data['targets'],p['targets'])}
    rows = []; summaries = []; deltas = []
    source_map = {m['name']:m for m in p['methods']}
    for method in p['methods']:
        current = []
        for item in p['targets']:
            with np.load(item['reference']) as data: reference = data['backbone'].copy()
            source = Path(method['directory'])/(item['id']+'.npz')
            with np.load(source) as data: pred = data[method['array']].copy()
            validate_pair(pred,reference,item['length'])
            current.append(dict(method=method['name'],seed=method['seed'],target=item['id'],subset=item['subset'],
                                metrics=metrics(pred,reference),source=str(source)))
        paired_rows(current,p['targets']); rows.extend(current)
        for subset in ['train','validation']:
            selected=[r for r in current if r['subset']==subset]
            aggregate={key:float(np.mean([r['metrics'][key] for r in selected])) for key in selected[0]['metrics']}
            if not all(np.isfinite(v) for v in aggregate.values()): raise RuntimeError('Undefined aggregate')
            summaries.append(dict(method=method['name'],seed=method['seed'],subset=subset,proteins=len(selected),metrics=aggregate))
            if method['name']!='author_fine':
                changes={}; counts={}
                for key in aggregate:
                    values=[r['metrics'][key]-original[r['target']]['metrics'][key] for r in selected]
                    changes[key]=float(np.mean(values));counts[key]=dict(lower=sum(v<0 for v in values),equal=sum(v==0 for v in values),higher=sum(v>0 for v in values))
                deltas.append(dict(method=method['name'],subset=subset,delta=changes,protein_direction_counts=counts))
    # Every unified CA/CN statistic must reproduce its already published development result.
    parity=[]
    for method in p['methods']:
        if method['name']=='author_fine': continue
        old=json.loads(Path(method['old_evaluation']).read_text())['targets']; old={r['target']:r for r in old}
        for row in [r for r in rows if r['method']==method['name']]:
            for key,old_key in [('ca_rmsd','rmsd'),('cn_rate','cn_rate'),('ca_clashes_per_residue','clashes_per_residue')]:
                if row['metrics'][key]!=old[row['target']]['metrics']['full'][old_key]: raise RuntimeError('Unified metric differs from prior '+method['name']+' '+row['target']+' '+key)
        parity.append(dict(method=method['name'],prior_ca_cn_clash_exact=True))
    result=dict(status='complete',scope=p['scope'],summaries=summaries,targets=rows,paired_deltas= deltas,
                author_parameters=author_data['parameters'],author_interface=author_data['interface'],
                cached_fine_total_differences=author_data['cached_fine_total_differences'],prior_metric_parity=parity,
                limitations=p['limitations'],protocol_sha256=sha(RUN/'protocol.json'))
    atomic_json(RUN/'comparison.json',result)
    lines=['作者原fine tokenizer与新两层AE：同输入重建对照','='*54,'',p['scope'],'',
           '两种方法都读取同一完整真实N/CA/C骨架。CA RMSD按全部CA独立Kabsch；backbone RMSD按全部3N原子独立Kabsch。没有删残基、选最好样本或用参考作回退。',
           '所有配置固定16k，两个种子分别报告。原作者模型不重复当作两个独立样本。每个蛋白等权。','',
           'method / subset / CA RMSD(Å) / backbone RMSD(Å) / CN(%) / boundary CN(%) / planarity(deg) / CA clashes per residue','']
    for row in summaries:
        m=row['metrics'];lines.append(f"{row['method']} / {row['subset']} / {m['ca_rmsd']:.4f} / {m['backbone_rmsd']:.4f} / {100*m['cn_rate']:.2f} / {100*m['boundary_cn_rate']:.2f} / {m['peptide_planarity_deviation_degrees']:.3f} / {m['ca_clashes_per_residue']:.5f}")
    lines+=['','原fine-only与完整decode最大坐标差：'+str(author_data['interface']['max_abs_error'])+'Å。',
            '重新编码与旧fine缓存不一致token总数：'+str(author_data['cached_fine_total_differences'])+'。',
            '作者模型参数量（含各分支）：'+json.dumps(author_data['parameters'])+'。',
            '新baseline/boundary186175参数，context/both191295参数。',
            '码率不同：作者每残基1个4096类整数（名义12bits）；新模型传输8ceil(N/4)+4N个连续浮点标量，未量化，不能声称同码率或压缩优势。',
            p['limitations'],'','逐蛋白配对差和方向计数见comparison.json；reference_metrics提供真实骨架自身几何基准。',
            '本次不训练、不调参、不选择最佳种子，不自动启动生成器或独立测试。']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in summaries if r['subset']=='validation']
    names=[r['method'] for r in chosen];colors=['#555555' if n=='author_fine' else '#2166ac' if n.startswith('context') else '#238b45' if n.startswith('both') else '#aaaaaa' for n in names]
    fig,axes=plt.subplots(1,3,figsize=(13,6),constrained_layout=True)
    for ax,(key,label,scale) in zip(axes,[('ca_rmsd','CA RMSD (Å)',1),('cn_rate','C–N outliers (%)',100),('peptide_planarity_deviation_degrees','Planarity deviation (degrees)',1)]):
        ax.bar(np.arange(len(chosen)),[r['metrics'][key]*scale for r in chosen],color=colors)
        ax.set_xticks(np.arange(len(chosen)),names,rotation=65,ha='right',fontsize=8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same 8 reused development structures · reconstruction only · unequal latent rate')
    fig.savefig(RUN/'comparison.png',dpi=180);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
    print('AUTHOR_TWO_LEVEL_COMPARISON_COMPLETE',json.dumps(chosen),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['author','report'],required=True);args=parser.parse_args()
    p=json.loads((RUN/'protocol.json').read_text());verify(p)
    if args.stage=='author': author(p)
    else: report(p)


if __name__=='__main__': main()
