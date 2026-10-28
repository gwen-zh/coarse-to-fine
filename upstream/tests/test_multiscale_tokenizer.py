import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from multiscale_tokenizer import pair_geometry, TwoScaleTokenizer, reconstruction_loss

class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        self.bb = torch.randn(24, 3, 3)
        self.z = torch.randn(24, 128)
        self.emb = torch.randn(24, 40)
    def test_rigid_motion_invariance(self):
        q, _ = torch.linalg.qr(torch.randn(3,3)); q[:,0] *= torch.linalg.det(q)
        a,m = pair_geometry(self.bb); b,n = pair_geometry(self.bb @ q + 12)
        torch.testing.assert_close(a,b,atol=8e-6,rtol=1e-5); self.assertTrue(torch.equal(m,n))
    def test_parameter_match_and_exact_initial_baseline(self):
        a,b = [TwoScaleTokenizer(m,decoder_dim=40).eval() for m in ('local','global')]
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        pair,mask=pair_geometry(self.bb)
        for model in (a,b):
            emb,ids,_=model(self.z,self.emb,pair,mask,torch.ones(40))
            self.assertTrue(torch.equal(emb,self.emb)); self.assertTrue(((ids>=0)&(ids<32)).all())
    def test_global_uses_edges_outside_local_mask(self):
        pair,mask=pair_geometry(self.bb);changed=pair.clone();changed[~mask]+=10
        local=TwoScaleTokenizer('local').eval(); glob=TwoScaleTokenizer('global').eval();glob.load_state_dict(local.state_dict())
        torch.testing.assert_close(local.encode(self.z,pair,mask),local.encode(self.z,changed,mask))
        self.assertGreater((glob.encode(self.z,pair,mask)-glob.encode(self.z,changed,mask)).abs().max().item(),1e-4)
    def test_hard_bottleneck_and_eval_no_mutation(self):
        model=TwoScaleTokenizer('global').eval();pair,mask=pair_geometry(self.bb)
        z=model.encode(self.z,pair,mask); before={k:v.clone() for k,v in model.quantizer.state_dict().items()}
        q,ids,loss=model.quantizer(z)
        torch.testing.assert_close(q,model.quantizer.codes[ids]);loss.backward()
        self.assertGreater(model.project.weight.grad.norm().item(),0)
        for k,v in model.quantizer.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
    def test_loss_rigid_invariance_and_gradient(self):
        q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
        loss,_=reconstruction_loss(self.bb@q+10,self.bb);self.assertLess(loss.item(),1e-8)
        x=(self.bb+torch.randn_like(self.bb)*.1).requires_grad_();loss,_=reconstruction_loss(x,self.bb);loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.norm().item(),0)

if __name__=='__main__':unittest.main()
