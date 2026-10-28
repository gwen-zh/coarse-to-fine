import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from multiscale_discrete import MultiScaleDiscrete,geometry_inputs,hard_embeddings,nominal_rate,retained_positions

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(8);self.bb=torch.randn(36,3,3)
 def test_same_parameters_rate(self):
  models=[MultiScaleDiscrete(m) for m in ['local','global','mixed']]
  self.assertEqual(len(set(sum(p.numel() for p in x.parameters()) for x in models)),1)
  self.assertEqual(nominal_rate(100),12);self.assertAlmostEqual(nominal_rate(101),12+6/101)
 def test_rigid_invariance(self):
  q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
  a,p,m=geometry_inputs(self.bb);b,r,n=geometry_inputs(self.bb@q+4)
  torch.testing.assert_close(a,b);torch.testing.assert_close(p,r,atol=1e-5,rtol=1e-5)
  for key in m:self.assertTrue(torch.equal(m[key],n[key]))
 def test_mask_isolates_pair_information(self):
  z,p,m=geometry_inputs(self.bb);model=MultiScaleDiscrete('mixed').eval();branch=model.branches[0];changed=p.clone();changed[~m['local']]+=5
  torch.testing.assert_close(branch.encode(z,p,m['local']),branch.encode(z,changed,m['local']))
  self.assertGreater((branch.encode(z,p,m['global'])-branch.encode(z,changed,m['global'])).abs().max().item(),1e-4)
 def test_quantized_interface_and_eval_immutable(self):
  model=MultiScaleDiscrete('mixed').eval();z,p,m=geometry_inputs(self.bb);before={k:v.clone() for k,v in model.state_dict().items()};q,ids,loss=model.encode(z,p,m)
  reconstructed=torch.cat([b.vq.codes[ids[:,i]] for i,b in enumerate(model.branches)],-1)
  torch.testing.assert_close(q,reconstructed);self.assertTrue(((ids>=0)&(ids<4)).all())
  for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
 def test_hidden_ids_cannot_enter_completer(self):
  model=MultiScaleDiscrete('mixed',vocab=8).eval();keep=retained_positions(8);ids=torch.zeros(8,dtype=torch.long);ids[~keep]=1
  with self.assertRaises(ValueError):model.complete(ids,keep,torch.zeros(8,48),torch.randn(8,128))
 def test_hard_forward_and_hidden_only_gradient(self):
  keep=retained_positions(8);visible=torch.zeros(8,dtype=torch.long);logits=torch.randn(8,12,requires_grad=True);table=torch.randn(12,7)
  emb,ids=hard_embeddings(logits,visible,keep,table);self.assertTrue(torch.equal(emb,table[ids]));self.assertTrue(torch.equal(ids[keep],visible[keep]))
  emb.square().mean().backward();self.assertEqual(float(logits.grad[keep].norm()),0);self.assertGreater(float(logits.grad[~keep].norm()),0)

if __name__=='__main__':unittest.main()
