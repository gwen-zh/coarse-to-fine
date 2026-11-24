from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from block_frame_tokenizer import BlockFrameFineTokenizer


class StableFineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_capacity_and_initial_weights_unchanged(self):
        torch.manual_seed(25);a=BlockFrameFineTokenizer('local');torch.manual_seed(25);b=StableBlockFrameFineTokenizer('local')
        self.assertEqual(sum(p.numel() for p in a.parameters()),193604)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
    def test_full_local_encoder_decoder_rigid_behavior(self):
        torch.manual_seed(28);model=StableBlockFrameFineTokenizer('local',fine_size=16).double().eval();centers=torch.randn(6,3,dtype=torch.float64);target=torch.randn(23,3,3,dtype=torch.float64);q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        a=model.context(centers,23);b=model.context(centers@q+3.,23);f=model.encode(target,a);g=model.encode(target@q+3.,b)
        torch.testing.assert_close(f,g,atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(model.decode_vectors(g,b,23),model.decode_vectors(f,a,23)@q+3.,atol=1e-8,rtol=1e-8)


if __name__=='__main__':unittest.main()
