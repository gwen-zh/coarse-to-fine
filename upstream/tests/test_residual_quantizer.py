import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from residual_quantizer import PrefixQuantizer, DecoderAdapter


class ResidualQuantizerTests(unittest.TestCase):
    def test_residual_codes_quantize_remainder(self):
        q = PrefixQuantizer(sizes=(2, 2), width=1)
        q.centers_0.copy_(torch.tensor([[0.], [10.]]))
        q.centers_1.copy_(torch.tensor([[0.], [2.]]))
        prefix, tokens = q(torch.tensor([[12.]]))
        self.assertEqual([t.item() for t in tokens], [1, 1])
        self.assertEqual([v.item() for v in prefix], [10., 12.])
        parallel = PrefixQuantizer(sizes=(2, 2), width=1, mode='parallel')
        parallel.load_state_dict(q.state_dict())
        values, _ = parallel(torch.tensor([[12.]]))
        self.assertEqual([v.item() for v in values], [10., 6.])

    def test_eval_does_not_update_and_ema_updates(self):
        q = PrefixQuantizer(sizes=(2, 3), width=2, decay=0.5)
        z = torch.tensor([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
        q.initialize(z)
        state = {k: v.clone() for k, v in q.state_dict().items()}
        q(z)
        self.assertTrue(all(torch.equal(v, q.state_dict()[k]) for k, v in state.items()))
        prefixes, _ = q(z, update=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in prefixes))
        self.assertFalse(torch.equal(state['counts_0'], q.counts_0))
        adapter = DecoderAdapter(width=2, hidden=4, output=3)
        sum(adapter(v).square().mean() for v in prefixes).backward()
        self.assertGreater(adapter.net[0].weight.grad.abs().sum().item(), 0)

    def test_unused_restart_uses_residuals(self):
        q = PrefixQuantizer(sizes=(1, 1), width=1)
        q.centers_0.fill_(10); q.counts_1.zero_()
        q.restart_unused(torch.tensor([[12.]]), seed=0)
        self.assertEqual(q.centers_1.item(), 2.)
        self.assertEqual(q.counts_1.item(), 1.)


if __name__ == '__main__':
    unittest.main()
