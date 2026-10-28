"""Conservative last-projection adaptation, leaving coarse/backbone untouched."""
import torch
from torch import nn
from torch.nn import functional as F

from coarse_lora import LoRALinear


def build_head(projection, rank=8):
    weight,bias=projection['weight'][:4096],projection['bias'][:4096]
    base=nn.Linear(weight.shape[1],len(weight))
    with torch.no_grad():
        base.weight.copy_(weight)
        base.bias.copy_(bias)
    return LoRALinear(base,rank=rank)


def retention_kl(student,teacher):
    """KL(teacher || student), averaged over residues, with no teacher gradient."""
    teacher=teacher.detach().float()
    return F.kl_div(student.float().log_softmax(-1),teacher.softmax(-1),reduction='batchmean')


def select_geometry_safe(candidates):
    baseline=next(r for r in candidates if r['epoch']==0)
    eligible=[baseline]
    for row in candidates:
        if row['epoch']==0:
            continue
        if (row['all_finite'] and row['ca_validity']>=baseline['ca_validity']-1e-12 and
                row['cn_rate']<=baseline['cn_rate']+1e-12 and
                ('ca_clash_rate' not in baseline or row.get('ca_clash_rate',float('inf'))<=baseline['ca_clash_rate']+1e-12)):
            eligible.append(row)
    return min(eligible,key=lambda r:(r['validation_ce'],r['epoch'])),eligible
