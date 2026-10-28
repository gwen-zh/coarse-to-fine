"""Reference-free acceptance for a frozen initial conformation."""
import math
import numpy as np
from structural_metrics import aligned_rmsd


def evaluate_candidate(tokens, backbone, initial_tokens, initial_backbone, metrics, baseline, delta, edit_cap):
    changed = float(np.mean(np.asarray(tokens) != np.asarray(initial_tokens)))
    displacement = aligned_rmsd(np.asarray(backbone)[:, 1], np.asarray(initial_backbone)[:, 1])
    cn, ca = metrics.get("backbone_cn_outliers"), metrics.get("upstream_ca_clashes")
    valid = (bool(metrics.get("all_ca_finite")) and bool(metrics.get("backbone_cn_finite"))
             and displacement is not None and math.isfinite(displacement)
             and cn is not None and ca is not None)
    eligible = bool(valid and displacement <= delta and changed <= edit_cap
                    and cn <= baseline["backbone_cn_outliers"] and ca <= baseline["upstream_ca_clashes"]
                    and (cn < baseline["backbone_cn_outliers"] or ca < baseline["upstream_ca_clashes"]))
    return dict(eligible=eligible, changed_fraction=changed, aligned_ca_displacement=displacement)
