import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from discrete_flow_math import status_probabilities,transition_probabilities,native_raw_weights,competing_exponential_sample

class FlowMathTests(unittest.TestCase):
    def setUp(self):
        self.x=torch.tensor([[2,0,1]]);p=torch.tensor([[[.25,.75,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float64);self.logp=p.log()
    def test_zero_corrector_exact_marginal(self):
        q=transition_probabilities(self.logp,self.x,torch.tensor([.4]),torch.tensor([.2]),0.,mask_id=2)
        torch.testing.assert_close(q[0,0],torch.tensor([.125,.375,.5],dtype=torch.float64))
        torch.testing.assert_close(q[0,1:],torch.eye(3,dtype=torch.float64)[:2])
    def test_native_normalized_matches_base_mask_row(self):
        mt=torch.tensor([.4],dtype=torch.float64);ms=torch.tensor([.2],dtype=torch.float64)
        raw=native_raw_weights(self.logp,mt,ms,mask_id=2);q=transition_probabilities(self.logp,self.x,mt,ms,0.,mask_id=2)
        torch.testing.assert_close(raw[0,0]/raw[0,0].sum(),q[0,0])
    def test_corrector_reintroduces_mask_and_is_normalized(self):
        q=transition_probabilities(self.logp,self.x,torch.tensor([.4]),torch.tensor([.2]),1.,mask_id=2)
        self.assertGreater(q[0,1,2].item(),0.);torch.testing.assert_close(q.sum(-1),torch.ones((1,3),dtype=torch.float64));self.assertGreaterEqual(q.min().item(),0.)
    def test_first_order_path_consistency_with_oracle(self):
        mt=torch.tensor([.4],dtype=torch.float64);ms=torch.tensor([.39999],dtype=torch.float64)
        q=transition_probabilities(self.logp,self.x,mt,ms,.5,mask_id=2)[0]
        # State order here is MASK,token0,token1.
        initial=torch.tensor([.4,.6*.25,.6*.75],dtype=torch.float64)
        final=initial@q;expected=torch.tensor([(1-.39999)*.25,(1-.39999)*.75,.39999],dtype=torch.float64)
        torch.testing.assert_close(final,expected,atol=1e-8,rtol=0.)
    def test_endpoint_and_identity_interval(self):
        a,b=status_probabilities(torch.tensor([.99999,.3]),torch.tensor([.00001,.3]),1.)
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all());self.assertEqual(a[1].item(),0.);self.assertEqual(b[1].item(),0.)
    def test_bad_schedule_rejected(self):
        with self.assertRaises(ValueError):status_probabilities(.2,.3,.5)
        with self.assertRaises(ValueError):status_probabilities(.2,.1,-1.)
    def test_native_rng_operation(self):
        weights=torch.rand(2,4,5);torch.manual_seed(12);a=competing_exponential_sample(weights)
        torch.manual_seed(12);denominator=1e-10-(torch.rand_like(weights)+1e-10).log();b=(weights/denominator).argmax(-1)
        torch.testing.assert_close(a,b)

if __name__=="__main__":unittest.main()
