import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
from joint_sequence_predictor import JointSequencePredictor,encode_aa,AA_ALPHABET
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_latent_contract import pool_ca

class JointSequencePredictorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        self.model=JointSequencePredictor(torch.zeros(16),torch.ones(16),width=32,layers=1,fine_size=16).eval()
        self.aa=encode_aa('ACDEFGHIKLMNPQRST')
    def test_ragged_and_short_output_shapes(self):
        for n in [1,4,5,17]:
            o=self.model(self.aa[:n]);self.assertEqual(o['coarse'].shape,((n+3)//4,16));self.assertEqual(o['fine_logits'].shape,(n,16));self.assertTrue(torch.isfinite(o['coarse']).all())
    def test_four_residues_share_coarse_but_keep_separate_states(self):
        c=torch.arange(3*16.).reshape(3,16);broadcast=self.model.broadcast_coarse(c,9)
        torch.testing.assert_close(broadcast,torch.cat([c[0:1].expand(4,-1),c[1:2].expand(4,-1),c[2:3]]))
        h=self.model.sequence_states(encode_aa('AAAAAAAAA'));self.assertFalse(torch.equal(h[0],h[1]))
    def test_fixed_sequence_states_changed_coarse_changes_fine_logits(self):
        h=self.model.sequence_states(self.aa);c=torch.zeros(5,16)
        a=self.model.fine_logits_from_states(h,c);b=self.model.fine_logits_from_states(h,c+torch.randn_like(c))
        self.assertFalse(torch.equal(a,b))
    def test_fine_ce_gradient_reaches_coarse_head_and_sequence_backbone(self):
        o=self.model(self.aa);loss=torch.nn.functional.cross_entropy(o['fine_logits'],torch.arange(17)%16);loss.backward()
        for name,module in [('coarse_head',self.model.coarse_head),('aa_embedding',self.model.aa_embedding),('transformer',self.model.transformer)]:
            norm=sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(norm,0,name)
    def test_coarse_target_is_label_not_fine_condition(self):
        f=torch.arange(17)%16;a=torch.zeros(5,16,requires_grad=True);b=torch.ones(5,16,requires_grad=True)
        loss1,t1=self.model.supervised_loss(self.aa,a,f);loss2,t2=self.model.supervised_loss(self.aa,b,f)
        self.assertTrue(torch.equal(t1['fine_ce'],t2['fine_ce']));self.assertFalse(torch.equal(t1['coarse_mse'],t2['coarse_mse']))
        loss1.backward();self.assertIsNone(a.grad)
    def test_output_replays_and_fits_frozen_decoder(self):
        base=HierarchicalRigidVQ(width=32,layers=1,coarse_size=8,fine_size=16);base.fine_book.initialize(torch.randn(128,8),2);decoder=HybridTokenizer(base);before=copy.deepcopy(decoder.state_dict())
        c,f=self.model.predict(self.aa);c2,f2=self.model.predict(self.aa)
        self.assertTrue(torch.equal(c,c2));self.assertTrue(torch.equal(f,f2));self.assertTrue((f<16).all())
        bb=decoder.decode(c,f);self.assertEqual(bb.shape,(17,3,3));torch.testing.assert_close(pool_ca(bb),decoder.coarse_centers(c,17),atol=2e-5,rtol=1e-6)
        for k,v in before.items():self.assertTrue(torch.equal(v,decoder.state_dict()[k]))
    def test_state_dict_restore_exact(self):
        other=JointSequencePredictor(torch.zeros(16),torch.ones(16),**self.model.config).eval();other.load_state_dict(self.model.state_dict())
        for key,value in self.model(self.aa).items():torch.testing.assert_close(value,other(self.aa)[key],atol=0,rtol=0)
    def test_invalid_inputs_rejected(self):
        for seq in ['', 'AA-G', 'AA?']:
            with self.assertRaises(ValueError):encode_aa(seq)
        with self.assertRaises(ValueError):self.model(torch.tensor([len(AA_ALPHABET)],dtype=torch.long))
        with self.assertRaises(ValueError):self.model.broadcast_coarse(torch.zeros(4,16),17)
        with self.assertRaises(ValueError):self.model.supervised_loss(self.aa,torch.zeros(5,16),torch.full((17,),16,dtype=torch.long))

if __name__=='__main__':unittest.main()
