import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from half_fine_decode_diagnostic import nearest_decoder_tokens,decode_variants

class DecodeDiagnosticTests(unittest.TestCase):
    def test_standardized_metric_and_original_token_roundtrip(self):
        table=torch.tensor([[0.,4.],[2.,0.]])
        # Raw nearest to origin is token1; after channel scaling nearest is0.
        self.assertEqual(int(nearest_decoder_tokens(torch.zeros(1,2),table,torch.tensor([1.,10.]))),0)
        torch.testing.assert_close(nearest_decoder_tokens(table,table,torch.tensor([1.,10.])),torch.arange(2))
    def test_visible_preservation_and_hidden_truth_not_read(self):
        torch.manual_seed(1);table=torch.randn(7,4);pred=torch.randn(6,4);original=torch.randn(6,4);keep=torch.arange(6)%2==0
        outputs,ids=decode_variants(pred,original,keep,table,torch.ones(4))
        other=original.clone();other[~keep]=float('nan')
        alternate,other_ids=decode_variants(pred,other,keep,table,torch.ones(4))
        self.assertTrue(torch.equal(ids,other_ids))
        for name in outputs:torch.testing.assert_close(outputs[name],alternate[name],rtol=0,atol=0)
        for name in ('visible_fixed','hidden_snapped'):
            self.assertTrue(torch.equal(outputs[name][keep],original[keep]))
        self.assertTrue(torch.equal(outputs['hidden_snapped'][~keep],table[ids]))
        self.assertTrue(torch.equal(outputs['visible_fixed'][~keep],pred[~keep]))
    def test_reject_wrong_latent_space_and_special_scale(self):
        with self.assertRaises(ValueError):nearest_decoder_tokens(torch.randn(2,128),torch.randn(7,1280),torch.ones(1280))
        with self.assertRaises(ValueError):nearest_decoder_tokens(torch.randn(2,4),torch.randn(7,4),torch.zeros(4))

if __name__=='__main__':unittest.main()
