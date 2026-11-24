import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
from joint_sequence_predictor import encode_aa
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow,snap_fine,fine_attention_mask
from hierarchical_rigid_vq import HierarchicalRigidVQ
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_latent_contract import pool_ca

class EnsembleFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.model=HierarchicalEnsembleFlow(torch.zeros(16),torch.ones(16),torch.zeros(8),torch.ones(8),width=32,layers=2).eval()
        self.aa=encode_aa('ACDEFGHIKLMNPQRST')
    def test_velocity_shapes_short_and_ragged(self):
        for n in [1,4,5,17]:
            aa=self.aa[:n];c=torch.randn((n+3)//4,16);f=torch.randn(n,8)
            self.assertEqual(self.model.coarse_velocity(aa,c,.4).shape,c.shape);self.assertEqual(self.model.fine_velocity(aa,f,.7,c).shape,f.shape)
    def test_seed_replay_diversity_and_global_rng_isolation(self):
        before=torch.get_rng_state().clone();a=self.model.sample_latents(self.aa,5,3,3);b=self.model.sample_latents(self.aa,5,3,3);c=self.model.sample_latents(self.aa,6,3,3)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for x,y,z in zip(a,b,c):self.assertTrue(torch.equal(x,y));self.assertFalse(torch.equal(x,z))
    def test_same_fine_seed_responds_to_sampled_coarse(self):
        c1=self.model.sample_coarse(self.aa,5,3);c2=self.model.sample_coarse(self.aa,6,3)
        f1=self.model.sample_fine(self.aa,c1,8,3);f2=self.model.sample_fine(self.aa,c2,8,3)
        self.assertFalse(torch.equal(f1,f2))
    def test_fine_seeds_do_not_modify_condition_coarse(self):
        c=self.model.sample_coarse(self.aa,5,3);before=c.clone()
        f1=self.model.sample_fine(self.aa,c,7,3);f2=self.model.sample_fine(self.aa,c,8,3)
        self.assertTrue(torch.equal(c,before));self.assertFalse(torch.equal(f1,f2))
    def test_coarse_context_cannot_attend_fine_state(self):
        c=torch.randn(5,16);f=torch.randn(17,8);a=self.model.fine_hidden(self.aa,f,.3,c);b=self.model.fine_hidden(self.aa,f+torch.randn_like(f),.3,c)
        torch.testing.assert_close(a[:5],b[:5],atol=1e-6,rtol=1e-6);self.assertFalse(torch.equal(a[5:],b[5:]))
        mask=fine_attention_mask(5,17,'cpu');self.assertTrue(mask[:5,5:].all());self.assertFalse(mask[5:,:5].any())
    def test_block_condition_retains_aa_order(self):
        a=encode_aa('ACDE');b=encode_aa('CADE')
        self.assertFalse(torch.equal(self.model.aa_conditions(a)[1],self.model.aa_conditions(b)[1]))
    def test_flow_loss_gradients_and_labels_not_trainable(self):
        c=torch.randn(5,16,requires_grad=True);f=torch.randn(17,8,requires_grad=True)
        loss,terms=self.model.flow_loss(self.aa,c,f,generator=torch.Generator().manual_seed(33));loss.backward()
        for module in [self.model.transformer,self.model.aa_embedding,self.model.coarse_head,self.model.fine_head,self.model.coarse_to_fine]:
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None),0)
        self.assertIsNone(c.grad);self.assertIsNone(f.grad);self.assertTrue(torch.isfinite(loss));self.assertEqual(len(terms['draw_sha256']),64)
    def test_jitter_comparison_pairs_noise_time_draws(self):
        c=torch.randn(5,16);f=torch.randn(17,8);draws=[]
        for sigma in [0.,.05]:
            _,terms=self.model.flow_loss(self.aa,c,f,coarse_jitter_std=sigma,generator=torch.Generator().manual_seed(31));draws.append(terms['draw_sha256'])
        self.assertEqual(draws[0],draws[1])
    def test_snap_uses_original_euclidean_code_lookup(self):
        codes=torch.randn(16,8);f=codes[[1,3,5]].clone();ids,error=snap_fine(f,codes)
        torch.testing.assert_close(ids,torch.tensor([1,3,5]));torch.testing.assert_close(error,torch.zeros(3))
    def test_frozen_decoder_and_center_contract(self):
        base=HierarchicalRigidVQ(width=32,layers=1,coarse_size=8,fine_size=16);base.fine_book.initialize(torch.randn(64,8),2);decoder=HybridTokenizer(base);before=copy.deepcopy(decoder.state_dict())
        out=self.model.sample_structure(self.aa,decoder,19,3,3)
        self.assertEqual(out['backbone'].shape,(17,3,3));self.assertTrue((out['fine_ids']<16).all())
        torch.testing.assert_close(pool_ca(out['backbone']),decoder.coarse_centers(out['coarse'],17),atol=2e-5,rtol=1e-6)
        for k,v in before.items():self.assertTrue(torch.equal(v,decoder.state_dict()[k]))
    def test_checkpoint_and_next_loss_replay(self):
        other=HierarchicalEnsembleFlow(torch.zeros(16),torch.ones(16),torch.zeros(8),torch.ones(8),**self.model.config).eval();other.load_state_dict(self.model.state_dict());c=torch.randn(5,16);f=torch.randn(17,8)
        a,_=self.model.flow_loss(self.aa,c,f,generator=torch.Generator().manual_seed(51));b,_=other.flow_loss(self.aa,c,f,generator=torch.Generator().manual_seed(51));torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):self.model.coarse_velocity(self.aa,torch.zeros(4,16),.2)
        with self.assertRaises(ValueError):self.model.fine_velocity(self.aa,torch.zeros(17,8),2.,torch.zeros(5,16))
        with self.assertRaises(ValueError):self.model.sample_latents(self.aa,1,0,2)
        with self.assertRaises(ValueError):snap_fine(torch.ones(3,8),torch.ones(4,16))

if __name__=='__main__':unittest.main()
