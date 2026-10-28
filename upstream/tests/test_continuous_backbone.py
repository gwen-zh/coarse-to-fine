import sys,types,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from continuous_backbone import ContinuousStateInput,ContinuousBackbone,ScopedLoRALinear

class Encoder(nn.Module):
    def __init__(self):super().__init__();self.embed=nn.Embedding(37,16)
    def forward(self,seq,coarse,scale):return self.embed(coarse)
class Heads(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(16,40)
    def forward(self,z=None,scale=None):return types.SimpleNamespace(structure_logits=self.linear(z))
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.transformer=nn.Module();self.transformer.blocks=nn.ModuleList([]);self.middle=nn.Linear(16,16);self.output_heads=Heads()
    def forward(self,var_structure_tokens,sequence_tokens,auxiliary_embeddings,lengths,scale):
        x=self.encoder(sequence_tokens,var_structure_tokens,scale);return self.output_heads(z=torch.tanh(self.middle(x))+x,scale=scale)

class ContinuousTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8);self.net=Net();self.wrapper=ContinuousBackbone(self.net,width=16,latent_dim=4,last_blocks=0);self.seq=torch.ones(1,7,dtype=torch.long);self.coarse=torch.tensor([[34,1,2,3,4,5,33]]);self.z=torch.randn(1,5,4);self.t=torch.tensor([.4])
    def test_disabled_exact_replay(self):
        a=self.net(self.coarse,self.seq,None,torch.tensor([5]),2).structure_logits[:,1:-1]
        b=self.wrapper(self.seq,self.coarse,self.z,self.t,enabled=False)['native_logits'];torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_state_changes_trunk_and_has_gradient(self):
        z=self.z.clone().requires_grad_();a=self.wrapper(self.seq,self.coarse,z,self.t);b=self.wrapper(self.seq,self.coarse,z+1,self.t)
        self.assertGreater((a['hidden']-b['hidden']).abs().max().item(),0)
        g=torch.autograd.grad(a['hidden'].square().mean(),z)[0];self.assertGreater(g.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
    def test_special_positions_unchanged(self):
        x=self.wrapper.input_adapter(self.z,self.t);self.assertEqual(x[:,0].abs().sum().item(),0);self.assertEqual(x[:,-1].abs().sum().item(),0)
    def test_zero_head_does_not_establish_upstream_gradient(self):
        a=self.wrapper(self.seq,self.coarse,self.z,self.t);g=torch.autograd.grad((a['velocity']-1).square().mean(),self.wrapper.input_adapter.state.weight)[0];self.assertEqual(g.abs().sum().item(),0)
    def test_hooks_cleaned_after_failure(self):
        self.z[0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.wrapper(self.seq,self.coarse,self.z,self.t)
        self.assertEqual(len(self.net.encoder._forward_hooks),0);self.assertFalse(self.wrapper._inside)
    def test_reject_batch_and_wrong_coarse(self):
        with self.assertRaises(ValueError):self.wrapper(self.seq.repeat(2,1),self.coarse.repeat(2,1),self.z.repeat(2,1,1),self.t.repeat(2))
        bad=self.coarse.clone();bad[0,2]=4095
        with self.assertRaises(ValueError):self.wrapper(self.seq,bad,self.z,self.t)
    def test_scoped_lora_disabled_preserves_base_after_change(self):
        layer=ScopedLoRALinear(nn.Linear(6,7),2);x=torch.randn(3,6)
        with torch.no_grad():layer.b.fill_(.1)
        torch.testing.assert_close(layer(x),layer.base(x),rtol=0,atol=0)
        layer.active=True;self.assertGreater((layer(x)-layer.base(x)).abs().max().item(),0)

if __name__=='__main__':unittest.main()
