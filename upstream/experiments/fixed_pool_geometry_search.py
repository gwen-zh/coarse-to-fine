"""Fixed-pool search with reference-free acceptance and exact decoder-call budget."""
import numpy as np
from structural_metrics import summarize_backbone
from fine_search_constraints import evaluate_candidate


def search_fixed_pool(original,pool,baseline,budget,decode,delta=1.,edit_cap=.1):
    original=np.asarray(original);pool=np.asarray(pool)
    if pool.ndim!=2 or pool.shape[0]!=len(original) or budget<0:raise ValueError("Invalid search inputs")
    if not all(len(set(row))==len(row) and int(token) in row for row,token in zip(pool,original)):raise ValueError("Pool must be unique and contain origin")
    base_metrics=summarize_backbone(baseline);current=original.copy();best_bb=baseline.copy();best_metrics=base_metrics;key=(base_metrics["backbone_cn_outliers"],base_metrics["upstream_ca_clashes"]);proposals=[];cursor=0;trace=[];padding=0
    for call in range(budget):
        if not proposals or cursor>=len(proposals):
            cn=np.linalg.norm(best_bb[:-1,2]-best_bb[1:,0],axis=-1);positions=set(np.where((cn<1)|(cn>1.8))[0].tolist());positions.update(i+1 for i in list(positions));ca=best_bb[:,1];x,y=np.where(np.triu(np.linalg.norm(ca[:,None]-ca[None],axis=-1)<3,1));positions.update(x.tolist());positions.update(y.tolist())
            proposals=[(site,int(token)) for site in sorted(positions) for token in pool[site] if token!=current[site]];cursor=0
        ids=current.copy();is_padding=not proposals
        if is_padding:padding+=1
        else:site,token=proposals[cursor];cursor+=1;ids[site]=token
        bb=decode(ids);metrics=summarize_backbone(bb);gate=evaluate_candidate(ids,bb,original,baseline,metrics,base_metrics,delta,edit_cap)
        candidate_key=(metrics.get("backbone_cn_outliers",10**9),metrics.get("upstream_ca_clashes",10**9));accepted=not is_padding and gate["eligible"] and candidate_key<key
        if accepted:current=ids.copy();best_bb=bb.copy();best_metrics=metrics;key=candidate_key;proposals=[]
        trace.append(dict(call=call+1,accepted=accepted,padding=is_padding,gate=gate,cn_count=metrics.get("backbone_cn_outliers"),ca_clashes=metrics.get("upstream_ca_clashes")))
    return current,best_bb,dict(selected=best_metrics,gate=evaluate_candidate(current,best_bb,original,baseline,best_metrics,base_metrics,delta,edit_cap),returned_original=bool(np.array_equal(current,original)),decoder_calls=len(trace),padding_calls=padding,trace=trace)
