"""Array transport and paired timing arithmetic without model namespace imports."""
import base64,io
import numpy as np


def pack(array):
    f=io.BytesIO();np.save(f,np.asarray(array),allow_pickle=False)
    return base64.b64encode(f.getvalue()).decode('ascii')


def unpack(value):
    return np.load(io.BytesIO(base64.b64decode(value)),allow_pickle=False)


def paired_summary(rows):
    if not rows:raise ValueError('Empty timing records')
    baseline=np.asarray([r['original_wall_seconds'] for r in rows],float)
    refined=np.asarray([r['refined_wall_seconds'] for r in rows],float)
    if not np.isfinite(baseline).all() or not np.isfinite(refined).all() or (baseline<=0).any() or (refined<=0).any():raise ValueError('Invalid timing')
    return dict(measured_pairs=len(rows),mean_original_seconds=float(baseline.mean()),mean_refined_seconds=float(refined.mean()),mean_added_seconds=float((refined-baseline).mean()),ratio_of_means=float(refined.mean()/baseline.mean()),added_percent=float(100*(refined.mean()/baseline.mean()-1)),median_paired_ratio=float(np.median(refined/baseline)))
