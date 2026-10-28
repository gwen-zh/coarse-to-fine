import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from fine_retention import build_head,retention_kl,select_geometry_safe


class FineRetentionTests(unittest.TestCase):
    def test_head_starts_at_original_logits_and_only_adapter_learns(self):
        projection=dict(weight=torch.randn(4101,4),bias=torch.randn(4101))
        model=build_head(projection,rank=2)
        h=torch.randn(3,4)
        expected=torch.nn.functional.linear(h,projection['weight'][:4096],projection['bias'][:4096])
        self.assertTrue(torch.equal(model(h),expected))
        model(h).square().mean().backward()
        self.assertIsNone(model.base.weight.grad)
        self.assertGreater(model.b.grad.abs().sum(),0)

    def test_kl_direction_residue_normalization_and_detached_teacher(self):
        teacher=torch.tensor([[2.,-1.],[0.,1.]],requires_grad=True)
        student=torch.tensor([[0.,1.],[1.,0.]],requires_grad=True)
        p=teacher.detach().softmax(-1)
        expected=(p*(p.log()-student.log_softmax(-1))).sum(-1).mean()
        loss=retention_kl(student,teacher)
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum(),0)
        self.assertLess(abs(retention_kl(teacher,teacher).item()),1e-6)

    def test_geometry_gate_rejects_lower_ce_with_worse_geometry(self):
        base=dict(epoch=0,validation_ce=2.,all_finite=True,ca_validity=.8,cn_rate=.1)
        worse=dict(base,epoch=1,validation_ce=.5,ca_validity=.7)
        safe=dict(base,epoch=3,validation_ce=1.5,cn_rate=.09)
        selected,eligible=select_geometry_safe([base,worse,safe])
        self.assertEqual(selected['epoch'],3)
        self.assertEqual([r['epoch'] for r in eligible],[0,3])
        self.assertEqual(select_geometry_safe([base,worse])[0]['epoch'],0)
        self.assertEqual(select_geometry_safe([dict(base,ca_clash_rate=.1),
            dict(safe,ca_clash_rate=.2)])[0]['epoch'],0)


if __name__=='__main__':unittest.main()
